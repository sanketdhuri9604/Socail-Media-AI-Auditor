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

def agent_audit(obs: dict) -> dict:
    """Call LLM to audit the social media post + AI analysis."""
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

    # Retry logic for rate limits
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
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = (attempt + 1) * 10
                print(json.dumps({"event": "RATE_LIMIT", "waiting_seconds": wait}))
                time.sleep(wait)
            else:
                raise e

    raise RuntimeError("Failed after 3 retries due to rate limits")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    episode_rewards = []
    start_time = time.time()

    # [START] log — mandatory format
    print(json.dumps({
        "event": "START",
        "env": "social_media_auditor_env",
        "model": MODEL_NAME,
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

        # Agent decides
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