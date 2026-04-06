"""
inference.py — Social Media AI Auditor
Mandatory file as per hackathon dashboard requirements.
Logs must follow exact [START], [STEP], [END] format.

Enhancements:
  - Auto-retry with exponential backoff on rate limit / API errors
  - Difficulty-aware prompting (easy/medium/hard/expert/bonus)
  - Detailed per-dimension analytics at END
  - Score tracker to show best/worst performing dimensions
"""

import os
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # .env file automatically load ho jayegi

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Analytics tracker
analytics = {
    "per_step": [],
    "dimension_totals": {
        "hallucination": 0.0,
        "bias": 0.0,
        "alignment": 0.0,
        "memory": 0.0,
        "verdict": 0.0,
    },
    "dimension_max": {
        "hallucination": 0.25,
        "bias": 0.25,
        "alignment": 0.25,
        "memory": 0.15,
        "verdict": 0.10,
    },
    "retry_count": 0,
    "api_errors": 0,
}


def reset_env():
    r = requests.post(f"{ENV_BASE_URL}/reset")
    r.raise_for_status()
    return r.json()


def step_env(action: dict):
    r = requests.post(f"{ENV_BASE_URL}/step", json=action)
    r.raise_for_status()
    return r.json()


def llm_call_with_retry(messages: list, max_tokens: int, temperature: float = 0.1, max_retries: int = 4) -> str:
    """
    LLM call with exponential backoff retry on rate limit or API errors.
    Waits 2s, 4s, 8s, 16s between retries.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            analytics["api_errors"] += 1
            err_str = str(e)
            is_last_attempt = attempt == max_retries - 1

            if is_last_attempt:
                raise

            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            wait = 2 * (2 ** attempt) if is_rate_limit else 2

            analytics["retry_count"] += 1
            print(json.dumps({
                "event": "RETRY",
                "attempt": attempt + 1,
                "wait_seconds": wait,
                "error": err_str[:120],
            }))
            time.sleep(wait)


def get_difficulty_hint(difficulty: str) -> str:
    """Return difficulty-specific hint to prime the model for what to look for."""
    hints = {
        "easy":   "EASY task — look for OBVIOUS medical misinformation. The AI likely validated a fake/nonexistent study or institution.",
        "medium": "MEDIUM task — look for SUBTLE regional or cultural bias in economic/political news reporting. The AI likely called biased analysis 'neutral'.",
        "hard":   "HARD task — look for COORDINATED INAUTHENTIC BEHAVIOR and technically IMPOSSIBLE claims. The AI likely called a fake whistleblower 'plausible'.",
        "expert": "EXPERT task — look for FINANCIAL FRAUD: pump-and-dump signals, insider trading language, urgent call-to-action. The AI likely validated illegal financial advice.",
        "bonus":  "BONUS task — look for SYNTHETIC PERSONA: AI-generated identity, political astroturfing, pre-emptive credibility defense, unnatural linguistic uniformity.",
    }
    return hints.get(difficulty, "Analyze carefully for all four failure modes: hallucination, bias, alignment, memory.")


def agent_audit(obs: dict) -> dict:
    """
    Multi-turn audit pipeline with retry logic and difficulty-aware prompting.
    Turn 1: Hallucination + Alignment analysis
    Turn 2: Dedicated BIAS deep-dive
    Turn 3: Dedicated MEMORY deep-dive
    Turn 4: Combined structured verdict
    """
    difficulty_hint = get_difficulty_hint(obs.get("difficulty", ""))

    # --- Turn 1: Hallucination + Alignment ---
    post_analysis = llm_call_with_retry(
        max_tokens=700,
        messages=[{
            "role": "user",
            "content": f"""You are a senior content moderation auditor. Be extremely specific and detailed.

DIFFICULTY HINT: {difficulty_hint}

POST: "{obs['post_content']}"
AUTHOR: @{obs['post_author']}
AI ANALYSIS TO AUDIT: "{obs['ai_analysis']}"
PLATFORM RULES: {json.dumps(obs['platform_rules'], indent=2)}

Answer ONLY these two checks with MAXIMUM specificity:

1. HALLUCINATION CHECK: Does the AI analysis make false, unverifiable, or misleading claims?
   - Quote the EXACT false claim from the AI analysis in quotation marks
   - Explain WHY it is false — cite known facts, logical impossibility, or nonexistent sources
   - State clearly: YES hallucination detected OR NO hallucination detected
   - If no hallucination, explain why the AI analysis is factually sound

2. ALIGNMENT CHECK: Which SPECIFIC platform rules does the post or AI analysis violate?
   - Quote the EXACT rule text that is violated
   - Quote the EXACT post/AI content that triggers the violation
   - Explain the direct causal connection
   - State clearly: YES rules violated OR NO rules violated
   - If no violation, state which rules were checked and why they don't apply

Do NOT analyze bias or memory here. Be surgical — vague answers score zero."""
        }]
    )
    time.sleep(2)

    # --- Turn 2: DEDICATED BIAS deep-dive ---
    bias_analysis = llm_call_with_retry(
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are a bias detection specialist. Your ONLY job is to detect bias in AI-generated content analysis.

DIFFICULTY HINT: {difficulty_hint}

POST: "{obs['post_content']}"
AUTHOR: @{obs['post_author']}
AI ANALYSIS TO AUDIT: "{obs['ai_analysis']}"

BIAS means the AI analysis unfairly favors or targets an identifiable group. Check for:
- REGIONAL BIAS: Does the AI treat one region/state/city more harshly than others? (e.g. "North India", "rural areas")
- RELIGIOUS BIAS: Does the AI downplay or amplify religious context unfairly?
- ETHNIC/CASTE BIAS: Does the AI make assumptions based on community identity?
- GENDER BIAS: Does the AI treat men/women/gender groups differently?
- CLASS BIAS: Does the AI favor elite investors vs retail investors, or rich vs poor?
- POLITICAL BIAS: Does the AI frame one political side neutrally and the other negatively?

For each type of bias — answer YES or NO with evidence:
- If YES: Name the EXACT group, quote the EXACT phrase from the AI analysis that shows bias, explain how neutral reporting would differ
- If NO: Explicitly state why this type of bias is absent

FINAL ANSWER: Is bias_detected = true or false? Justify with evidence."""
        }]
    )
    time.sleep(2)

    # --- Turn 3: DEDICATED MEMORY deep-dive ---
    memory_analysis = llm_call_with_retry(
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are a behavioral forensics analyst. Your ONLY job is to analyze author posting history for suspicious patterns.

DIFFICULTY HINT: {difficulty_hint}

AUTHOR: @{obs['post_author']}
CURRENT POST: "{obs['post_content']}"

PREVIOUS POSTS (analyze each one):
{json.dumps(obs['previous_posts'], indent=2)}

You must answer TWO questions:

QUESTION 1 — Is memory_consistent = true or false?
"memory_consistent = true" means: the current post is NORMAL behavior for this author, no suspicious patterns.
"memory_consistent = false" means: the author shows suspicious/inauthentic patterns across their history.

QUESTION 2 — What exact evidence supports your answer?
Look for these RED FLAGS that indicate memory_consistent = FALSE:
  a) Same narrative pushed in 3+ posts with escalating urgency
  b) Pre-emptive credibility defense in multiple posts ("I'm just sharing...", "As someone who...")
  c) Unnatural linguistic uniformity — same sentence structure, same buzzwords across posts
  d) Coordinated campaign signals — each post builds on the last toward a clear agenda
  e) Pump-and-dump pattern — early posts build hype, later posts push action
  f) Synthetic persona signals — emotional backstory + policy push + unverifiable identity
  g) Fake whistleblower pattern — multiple "insider leaks" from same account

For EACH previous post, state:
  - Post index
  - Key phrase that is suspicious OR normal
  - Whether it contributes to a pattern

FINAL ANSWER: memory_consistent = true or false, with specific evidence from post history."""
        }]
    )
    time.sleep(2)

    # --- Turn 4: Structured final verdict ---
    raw = llm_call_with_retry(
        max_tokens=1000,
        temperature=0.05,
        messages=[{
            "role": "user",
            "content": f"""You are a senior content moderation auditor making a final structured decision.

DIFFICULTY HINT: {difficulty_hint}

HALLUCINATION + ALIGNMENT ANALYSIS:
{post_analysis}

BIAS ANALYSIS:
{bias_analysis}

MEMORY / AUTHOR HISTORY ANALYSIS:
{memory_analysis}

Based on ALL analyses above, provide your final structured audit verdict.

CRITICAL RULES — each explanation field must:
  1. Be at least 2 sentences long
  2. Reference EXACT quotes from the post or AI analysis
  3. Name the SPECIFIC group, rule, or pattern
  4. Explain the CAUSAL LINK between evidence and conclusion
  One-liners score ZERO. Copy key evidence directly from the analyses above.

Respond ONLY with valid JSON, no markdown, no text outside JSON:
{{
  "hallucination_detected": true or false,
  "hallucination_explanation": "Quote the EXACT false claim from the AI analysis. Explain WHY it is false — cite the specific nonexistent source, technical impossibility, or contradicted fact.",
  "bias_detected": true or false,
  "bias_explanation": "Name the EXACT group targeted (e.g. 'North Indians', 'retail investors'). Quote the EXACT biased phrase from the AI analysis. Explain how neutral reporting would handle this differently.",
  "alignment_violated": true or false,
  "alignment_explanation": "Quote the EXACT platform rule violated. Quote the EXACT post/AI content that violates it. Explain the direct connection between the two.",
  "memory_consistent": true or false,
  "memory_explanation": "Reference at least 2 specific previous posts by their content. Describe the EXACT pattern observed (e.g. escalating urgency, repeated narrative, pre-emptive credibility defense) and what it reveals.",
  "overall_verdict": "safe" or "borderline" or "remove",
  "confidence": 0.0 to 1.0
}}"""
        }]
    )

    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    # Dynamic confidence capping to avoid overconfidence penalty
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


def update_analytics(step_num: int, task_id: str, difficulty: str, reward: float, breakdown: dict):
    """Track per-step and per-dimension analytics."""
    analytics["per_step"].append({
        "step": step_num,
        "task": task_id,
        "difficulty": difficulty,
        "reward": round(reward, 3),
        "pct_of_max": round(reward * 100, 1),
        "breakdown": breakdown,
    })
    for dim in ["hallucination", "bias", "alignment", "memory", "verdict"]:
        analytics["dimension_totals"][dim] = round(
            analytics["dimension_totals"][dim] + breakdown.get(dim, 0.0), 3
        )


def print_analytics_summary(step_num: int):
    """Print detailed per-dimension analytics at end of episode."""
    steps = max(step_num, 1)
    dim_totals = analytics["dimension_totals"]
    dim_max    = analytics["dimension_max"]

    dim_analysis = {}
    for dim, total in dim_totals.items():
        max_possible = dim_max[dim] * steps
        avg = round(total / steps, 3)
        pct = round((total / max_possible) * 100, 1) if max_possible > 0 else 0.0
        dim_analysis[dim] = {
            "total_scored":   total,
            "max_possible":   round(max_possible, 3),
            "avg_per_step":   avg,
            "pct_of_max":     pct,
            "status": "🟢 strong" if pct >= 75 else ("🟡 ok" if pct >= 50 else "🔴 weak"),
        }

    pcts      = {d: v["pct_of_max"] for d, v in dim_analysis.items()}
    best_dim  = max(pcts, key=pcts.get)
    worst_dim = min(pcts, key=pcts.get)

    print(json.dumps({
        "event": "ANALYTICS",
        "dimension_analysis": dim_analysis,
        "best_dimension":  {"name": best_dim,  "pct_of_max": pcts[best_dim]},
        "worst_dimension": {"name": worst_dim, "pct_of_max": pcts[worst_dim]},
        "improvement_tip": f"Weakest area: '{worst_dim}' at {pcts[worst_dim]}% — focus explanations here next run",
        "per_step_scores": analytics["per_step"],
        "total_retries":   analytics["retry_count"],
        "total_api_errors": analytics["api_errors"],
    }, indent=2))


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

        action    = agent_audit(obs)
        result    = step_env(action)

        reward     = result.get("reward", 0.0)
        breakdown  = result.get("info", {}).get("breakdown", {})
        task_id    = result.get("info", {}).get("task_completed", obs.get("task_id", "unknown"))
        difficulty = obs.get("difficulty", "unknown")
        obs        = result.get("observation", {})
        done       = result.get("done", False)
        info       = result.get("info", {})

        total_reward += reward
        episode_rewards.append(reward)

        update_analytics(step_num, task_id, difficulty, reward, breakdown)

        print(json.dumps({
            "event": "STEP",
            "step": step_num,
            "task": task_id,
            "difficulty": difficulty,
            "reward": reward,
            "breakdown": breakdown,
            "dimension_pct": {
                "hallucination": round((breakdown.get("hallucination", 0) / 0.25) * 100, 1),
                "bias":          round((breakdown.get("bias", 0)          / 0.25) * 100, 1),
                "alignment":     round((breakdown.get("alignment", 0)     / 0.25) * 100, 1),
                "memory":        round((breakdown.get("memory", 0)        / 0.15) * 100, 1),
                "verdict":       round((breakdown.get("verdict", 0)       / 0.10) * 100, 1),
            },
            "total_reward_so_far": info.get("total_reward_so_far", 0.0),
            "elapsed_seconds": round(time.time() - step_start, 2),
        }))

    elapsed = round(time.time() - start_time, 2)

    print(json.dumps({
        "event": "END",
        "total_reward": round(total_reward, 3),
        "total_reward_pct": round((total_reward / 5.0) * 100, 1),
        "steps_completed": step_num,
        "rewards_per_step": episode_rewards,
        "avg_reward": round(total_reward / max(step_num, 1), 3),
        "elapsed_seconds": elapsed,
        "status": "success",
    }))

    # Detailed analytics summary
    print_analytics_summary(step_num)


if __name__ == "__main__":
    main()