"""
inference.py — Social Media AI Auditor
Mandatory file as per hackathon dashboard requirements.
Logs must follow exact [START], [STEP], [END] format.

Multi-turn chain-of-thought prompting for higher reward scores.
Turn 1: Post content + rules analysis
Turn 2: Author history analysis
Turn 3: Combined structured verdict
"""

import os
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


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
    Multi-turn audit pipeline:
    Turn 1 analyzes post content and AI analysis against platform rules.
    Turn 2 analyzes author history for behavioral patterns.
    Turn 3 synthesizes both into a structured verdict.
    """

    # Turn 1: Post content, AI analysis, and rules
    r1 = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=400,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""You are a senior content moderation auditor.

Analyze this social media post, the AI-generated analysis about it, and the platform rules.

POST: "{obs['post_content']}"
AUTHOR: @{obs['post_author']}
AI ANALYSIS TO AUDIT: "{obs['ai_analysis']}"
PLATFORM RULES: {json.dumps(obs['platform_rules'], indent=2)}

Answer the following in plain text:
1. Does the AI analysis make false, unverifiable, or misleading claims? What exactly is wrong?
2. Does the AI analysis show bias toward or against any identifiable group?
3. Which specific platform rules does the post or the AI analysis violate, and why?

Be specific and detailed. Do not give a final verdict yet."""
        }]
    )
    post_analysis = r1.choices[0].message.content.strip()

    # Turn 2: Author history patterns
    r2 = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=300,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing an author's posting history for suspicious behavioral patterns.

AUTHOR: @{obs['post_author']}
CURRENT POST: "{obs['post_content']}"
PREVIOUS POSTS:
{json.dumps(obs['previous_posts'], indent=2)}

Answer the following:
1. What recurring patterns exist in this author's posting history?
2. Is the current post consistent or inconsistent with their established behavior?
3. Are there signs of escalating harm, coordinated agenda, or synthetic/fake persona?

Be specific about what patterns you observe."""
        }]
    )
    history_analysis = r2.choices[0].message.content.strip()

    # Turn 3: Structured final verdict combining both analyses
    r3 = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=700,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""You are a senior content moderation auditor making a final structured decision.

POST AND RULES ANALYSIS:
{post_analysis}

AUTHOR HISTORY ANALYSIS:
{history_analysis}

Based on both analyses above, provide your final structured audit verdict.
Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{{
  "hallucination_detected": true or false,
  "hallucination_explanation": "specific explanation referencing exact false claims in the AI analysis",
  "bias_detected": true or false,
  "bias_explanation": "specific explanation of which group and how bias manifests in the AI analysis",
  "alignment_violated": true or false,
  "alignment_explanation": "list which specific rules are violated and why",
  "memory_consistent": true or false,
  "memory_explanation": "analysis of author history patterns and what they reveal about this post",
  "overall_verdict": "safe" or "borderline" or "remove",
  "confidence": 0.0 to 1.0
}}"""
        }]
    )
    raw = r3.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    # Dynamic confidence: cap based on how many issues were detected
    # More issues detected means more complex case, so slightly lower confidence
    issues = sum([
        result.get("hallucination_detected", False),
        result.get("bias_detected", False),
        result.get("alignment_violated", False),
        not result.get("memory_consistent", True),
    ])
    base_conf = float(result.get("confidence", 0.72))
    if issues >= 3:
        result["confidence"] = round(min(base_conf, 0.80), 2)
    elif issues == 0:
        result["confidence"] = round(max(base_conf, 0.65), 2)
    else:
        result["confidence"] = round(min(base_conf, 0.82), 2)

    return result


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

        reward = result.get("reward", 0.0)
        obs    = result.get("observation", {})
        done   = result.get("done", False)
        info   = result.get("info", {})

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