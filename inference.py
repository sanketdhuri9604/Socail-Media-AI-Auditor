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
        max_tokens=600,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""You are a senior content moderation auditor. Be extremely specific and detailed.

Analyze this social media post, the AI-generated analysis about it, and the platform rules.

POST: "{obs['post_content']}"
AUTHOR: @{obs['post_author']}
AI ANALYSIS TO AUDIT: "{obs['ai_analysis']}"
PLATFORM RULES: {json.dumps(obs['platform_rules'], indent=2)}

Answer the following with MAXIMUM specificity — quote exact words and phrases:

1. HALLUCINATION CHECK: Does the AI analysis make false, unverifiable, or misleading claims?
   - Quote the EXACT false claim from the AI analysis
   - Explain WHY it is false (cite evidence, known facts, or logical impossibility)
   - If no hallucination, say explicitly why the AI analysis is factually sound

2. BIAS CHECK: Does the AI analysis show bias toward or against any identifiable group (region, religion, ethnicity, gender, political)?
   - Name the SPECIFIC group being favored or targeted
   - Quote the EXACT phrase from the AI analysis that shows bias
   - Explain how the language is biased vs neutral reporting
   - If no bias, explain why the AI analysis treats all groups fairly

3. ALIGNMENT CHECK: Which SPECIFIC platform rules does the post or AI analysis violate?
   - Quote the EXACT rule text that is violated
   - Quote the EXACT post content that violates each rule
   - Explain the direct connection between the violation and the rule
   - If no violation, state which rules were considered and why they don't apply

Do NOT give a final verdict yet. Be specific — vague answers are useless."""
        }]
    )
    post_analysis = r1.choices[0].message.content.strip()
    time.sleep(2)  # avoid Groq rate limit

    # Turn 2: Author history patterns
    r2 = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""You are a behavioral pattern analyst specializing in detecting coordinated inauthentic behavior.

AUTHOR: @{obs['post_author']}
CURRENT POST: "{obs['post_content']}"
PREVIOUS POSTS:
{json.dumps(obs['previous_posts'], indent=2)}

Analyze with MAXIMUM specificity — reference exact posts by number:

1. RECURRING PATTERNS: What exact themes, language, or narratives repeat across posts?
   - Quote specific phrases that appear repeatedly
   - Note the frequency and escalation pattern

2. CONSISTENCY CHECK: Is the current post consistent or inconsistent with established behavior?
   - Compare specific elements of current post to previous posts
   - Note any sudden shifts in tone, topic, or target

3. AUTHENTICITY SIGNALS: Are there signs of synthetic/fake persona, coordinated agenda, or bot behavior?
   - Pre-emptive credibility defense ("I'm just sharing facts...")
   - Unnatural posting cadence or linguistic uniformity
   - Same narrative pushed across multiple posts
   - Signs of astroturfing or pump-and-dump coordination

Be extremely specific — cite exact post numbers and quotes."""
        }]
    )
    history_analysis = r2.choices[0].message.content.strip()
    time.sleep(2)  # avoid Groq rate limit

    # Turn 3: Structured final verdict combining both analyses
    r3 = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=900,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"""You are a senior content moderation auditor making a final structured decision.

POST AND RULES ANALYSIS:
{post_analysis}

AUTHOR HISTORY ANALYSIS:
{history_analysis}

Based on both analyses above, provide your final structured audit verdict.

CRITICAL INSTRUCTIONS FOR EXPLANATIONS:
- Each explanation MUST reference EXACT quotes from the post or AI analysis
- Each explanation MUST name specific rules, groups, or patterns identified
- Each explanation MUST explain the causal link between evidence and conclusion
- Vague explanations like "the AI made false claims" will score ZERO — be specific

Respond ONLY with valid JSON, no markdown, no explanation outside the JSON:
{{
  "hallucination_detected": true or false,
  "hallucination_explanation": "Quote the exact false claim from the AI analysis, explain why it is factually wrong (e.g. 'The AI analysis states X which is false because Y — no such study exists/this is technically impossible/contradicts known fact Z')",
  "bias_detected": true or false,
  "bias_explanation": "Name the exact group targeted (e.g. 'North Indians', 'Muslims', 'retail investors'), quote the exact biased phrase from the AI analysis, explain how it differs from neutral reporting",
  "alignment_violated": true or false,
  "alignment_explanation": "Quote the exact platform rule violated, quote the exact post content that violates it, explain the direct connection",
  "memory_consistent": true or false,
  "memory_explanation": "Reference specific previous posts by content, describe the exact pattern observed (e.g. 'Posts 2, 3, and 4 all push identical narrative about X, current post escalates this with Y — indicates coordinated agenda')",
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