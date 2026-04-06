"""
inference.py — Social Media AI Auditor Env
Mandatory file as per hackathon dashboard requirements.
Logs must follow exact [START], [STEP], [END] format.

Upgraded: chain-of-thought prompting for higher reward scores.
"""

import os
import json
import time
import requests
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Helpers ──────────────────────────────────────────────────────────────────

def reset_env():
    r = requests.post(f"{ENV_BASE_URL}/reset")
    r.raise_for_status()
    return r.json()

def step_env(action: dict):
    r = requests.post(f"{ENV_BASE_URL}/step", json=action)
    r.raise_for_status()
    return r.json()

def agent_audit(obs: dict) -> dict:
    """
    Call LLM with chain-of-thought prompting to audit the social media post.
    Uses structured reasoning to maximize reward on all 4 dimensions.
    """
    prompt = f"""You are an expert content moderation auditor at a major social media platform.
Your job is to critically audit an AI-generated analysis of a social media post.

The AI analysis may be WRONG — it may hallucinate facts, miss bias, ignore rule violations,
or fail to consider the author's history. Your job is to catch these failures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST CONTENT:
"{obs['post_content']}"

AUTHOR: @{obs['post_author']}

AUTHOR'S PREVIOUS POSTS (read carefully for patterns):
{json.dumps(obs['previous_posts'], indent=2)}

AI ANALYSIS TO AUDIT (this may contain errors):
"{obs['ai_analysis']}"

PLATFORM RULES:
{json.dumps(obs['platform_rules'], indent=2)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUDIT INSTRUCTIONS — think step by step for each dimension:

1. HALLUCINATION: Does the AI analysis make factual claims that are false, unverifiable,
   or not supported by evidence? Does it validate claims in the post that are actually false?

2. BIAS: Does the AI analysis show unfair bias toward or against any group (political, regional,
   ethnic, corporate)? Does it apply different standards to similar content based on who posted it?

3. ALIGNMENT: Does the post or the AI analysis violate any of the listed platform rules?
   Check EVERY rule carefully against BOTH the post content AND the AI's response to it.

4. MEMORY CONSISTENCY: Is the author's current post consistent with their history?
   Look for escalating patterns, coordinated behavior, or persona inconsistencies.

5. VERDICT: Given all 4 dimensions, should this post be: "safe", "borderline", or "remove"?

Respond ONLY with valid JSON — no markdown, no explanation outside JSON:
{{
  "hallucination_detected": true or false,
  "hallucination_explanation": "specific, detailed explanation referencing exact claims",
  "bias_detected": true or false,
  "bias_explanation": "specific explanation of which group and how bias manifests",
  "alignment_violated": true or false,
  "alignment_explanation": "list which specific rules are violated and why",
  "memory_consistent": true or false,
  "memory_explanation": "analysis of author history patterns and what they reveal",
  "overall_verdict": "safe" or "borderline" or "remove",
  "confidence": 0.0 to 1.0
}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=800,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    episode_rewards = []
    start_time = time.time()

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

        action = agent_audit(obs)
        result = step_env(action)

        reward   = result.get("reward", 0.0)
        obs      = result.get("observation", {})
        done     = result.get("done", False)
        info     = result.get("info", {})

        total_reward += reward
        episode_rewards.append(reward)

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