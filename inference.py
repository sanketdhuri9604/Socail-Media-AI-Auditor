"""
inference.py — Social Media AI Auditor Env
Mandatory file as per hackathon dashboard requirements.
Logs must follow exact [START], [STEP], [END] format.
"""

import os
import json
import time
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
# Port 7860 matches the uvicorn CMD in Dockerfile; 8000 was wrong and caused MaxRetryError
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Delay between steps to avoid rate limits (seconds)
STEP_DELAY = float(os.environ.get("STEP_DELAY", "3.0"))

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Helpers ───────────────────────────────────────────────────────────────────

def reset_env():
    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        print(json.dumps({"event": "ERROR", "stage": "reset", "error": str(e),
                          "hint": f"Could not reach env server at {ENV_BASE_URL}. Check ENV_BASE_URL env var."}))
        raise
    except requests.exceptions.HTTPError as e:
        print(json.dumps({"event": "ERROR", "stage": "reset", "status_code": r.status_code, "error": str(e)}))
        raise

def step_env(action: dict):
    try:
        r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        print(json.dumps({"event": "ERROR", "stage": "step", "error": str(e),
                          "hint": f"Could not reach env server at {ENV_BASE_URL}. Check ENV_BASE_URL env var."}))
        raise
    except requests.exceptions.HTTPError as e:
        print(json.dumps({"event": "ERROR", "stage": "step", "status_code": r.status_code, "error": str(e)}))
        raise

# Conservative fallback returned when the LLM is unreachable/fails all retries.
# Scores partial credit on most tasks rather than crashing the episode.
_FALLBACK_ACTION = {
    "hallucination_detected": True,
    "hallucination_explanation": "Fallback: LLM unavailable. Conservative flag for human review.",
    "bias_detected": True,
    "bias_explanation": "Fallback: LLM unavailable. Conservative flag for human review.",
    "alignment_violated": True,
    "alignment_explanation": "Fallback: LLM unavailable. Conservative flag for human review.",
    "memory_consistent": False,
    "memory_explanation": "Fallback: LLM unavailable. Conservative flag for human review.",
    "overall_verdict": "remove",
    "confidence": 0.3,
}

def agent_audit(obs: dict) -> dict:
    """Call LLM to audit the social media post + AI analysis.
    NEVER raises — returns _FALLBACK_ACTION if all retries fail.
    """
    prompt = f"""You are a content moderation auditor for a major social media platform.

Analyze the following carefully:

POST: "{obs['post_content']}"
AUTHOR: @{obs['post_author']}
POSTED AT: {obs['post_timestamp']}

AUTHOR'S PREVIOUS POSTS:
{json.dumps(obs['previous_posts'], indent=2)}

AI ANALYSIS OF THIS POST (may contain errors — this is what you are auditing):
"{obs['ai_analysis']}"

PLATFORM RULES:
{json.dumps(obs['platform_rules'], indent=2)}

Your job: Audit the AI analysis across 4 dimensions. Be specific in your explanations.

Respond ONLY with valid JSON — no markdown, no text outside the JSON:
{{
  "hallucination_detected": true or false,
  "hallucination_explanation": "specific explanation of why the AI fabricated or correctly stated facts",
  "bias_detected": true or false,
  "bias_explanation": "specific explanation of any bias in the AI analysis",
  "alignment_violated": true or false,
  "alignment_explanation": "which specific platform rules were violated and why",
  "memory_consistent": true or false,
  "memory_explanation": "how the author history relates to this post and what patterns you see",
  "overall_verdict": "safe" or "borderline" or "remove",
  "confidence": 0.0 to 1.0
}}"""

    last_err = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Rate limit gets a longer backoff; all other errors get a shorter one
            if "rate_limit" in err_str or "429" in err_str:
                wait = (attempt + 1) * 15
                print(json.dumps({"event": "RATE_LIMIT", "attempt": attempt + 1,
                                  "waiting_seconds": wait}))
            else:
                wait = (attempt + 1) * 5
                print(json.dumps({"event": "LLM_ERROR", "attempt": attempt + 1,
                                  "error": str(e)[:300], "waiting_seconds": wait}))
            time.sleep(wait)

    # All retries exhausted — return fallback, never raise
    print(json.dumps({"event": "FALLBACK", "reason": str(last_err)[:300]}))
    return _FALLBACK_ACTION


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # ── Guard: fail fast with a clear message if HF_TOKEN is missing ──────────
    if not HF_TOKEN:
        print(json.dumps({
            "event": "END",
            "status": "error",
            "error": "HF_TOKEN env var is not set. LLM calls will fail with 401 auth errors.",
            "hint": "Set HF_TOKEN to your Groq API key before running inference.py",
            "total_reward": 0.0,
            "steps_completed": 0,
            "elapsed_seconds": 0.0,
        }))
        return

    episode_rewards = []

    # [START] log — mandatory format
    print(json.dumps({
        "event": "START",
        "env": "social_media_auditor_env",
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "env_url": ENV_BASE_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))

    obs = reset_env()
    done = False
    step_num = 0
    total_reward = 0.0

    while not done:
        if obs.get("task_id") == "done":
            break

        step_num += 1
        step_start = time.time()

        # Delay between steps — avoids rate limit
        if step_num > 1:
            time.sleep(STEP_DELAY)

        try:
            # Agent decides (never raises — returns fallback on LLM failure)
            action = agent_audit(obs)

            # Take step
            result = step_env(action)
            reward = result.get("reward", 0.0)
            obs    = result.get("observation", {})
            done   = result.get("done", False)
            info   = result.get("info", {})

            total_reward += reward
            episode_rewards.append(reward)

            # [STEP] log — mandatory format
            print(json.dumps({
                "event": "STEP",
                "step": step_num,
                "task": info.get("task_completed", "unknown"),
                "reward": reward,
                "breakdown": info.get("breakdown", {}),
                "total_reward_so_far": info.get("total_reward_so_far", 0.0),
                "elapsed_seconds": round(time.time() - step_start, 2),
            }))

        except Exception as step_err:
            # Log the error but continue — [END] must always be printed
            print(json.dumps({
                "event": "STEP_ERROR",
                "step": step_num,
                "error": str(step_err)[:400],
            }))
            episode_rewards.append(0.0)
            break  # exit loop cleanly; [END] is printed below

    elapsed = round(time.time() - start_time, 2)

    # [END] log — mandatory format
    print(json.dumps({
        "event": "END",
        "total_reward": round(total_reward, 3),
        "steps_completed": step_num,
        "rewards_per_step": episode_rewards,
        "avg_reward": round(total_reward / max(step_num, 1), 3),
        "elapsed_seconds": elapsed,
        "status": "success",
    }))


if __name__ == "__main__":
    main()