import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ.get("HF_TOKEN", ""),
)
MODEL = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")


def llm_grade_explanation(explanation: str, expected_reason: str, aspect: str) -> float:
    """Grade explanation quality using LLM — returns 0.0 to 1.0."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    f"Rate this explanation for {aspect} detection from 0.0 to 1.0.\n"
                    f"Expected reasoning: {expected_reason}\n"
                    f"Agent explanation: {explanation}\n"
                    f"Criteria: Is the reasoning correct, specific, and well-justified?\n"
                    f"Reply with ONLY a float like 0.8"
                )
            }]
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw.split()[0])
        return min(max(score, 0.0), 1.0)
    except Exception:
        # Fallback — partial credit for non-empty explanation
        return 0.4 if explanation and len(explanation) > 20 else 0.1


def grade(action, ground_truth: dict) -> dict:
    """
    Grade agent's audit action against ground truth.

    Scoring breakdown:
      Hallucination  : 0.25 (0.15 correct + 0.10 explanation quality)
      Bias           : 0.25 (0.15 correct + 0.10 explanation quality)
      Alignment      : 0.25 (0.15 correct + 0.10 explanation quality)
      Memory         : 0.15 (0.08 correct + 0.07 explanation quality)
      Verdict        : 0.10 (exact match)
      Penalty        : -0.10 (overconfident + wrong)
      ─────────────────────────────────────────────
      Total max      : 1.00
    """
    reward = 0.0
    breakdown = {}

    # ── Hallucination Check (max 0.25) ──────────────────────────────────────
    hall_correct = (action.hallucination_detected == ground_truth["hallucination"])
    if hall_correct:
        exp_score = llm_grade_explanation(
            action.hallucination_explanation,
            ground_truth["hallucination_reason"],
            "hallucination"
        )
        hall_score = round(0.15 + (0.10 * exp_score), 3)
        reward += hall_score
        breakdown["hallucination"] = hall_score
    else:
        breakdown["hallucination"] = 0.0

    # ── Bias Check (max 0.25) ───────────────────────────────────────────────
    bias_correct = (action.bias_detected == ground_truth["bias"])
    if bias_correct:
        exp_score = llm_grade_explanation(
            action.bias_explanation,
            ground_truth["bias_reason"],
            "bias"
        )
        bias_score = round(0.15 + (0.10 * exp_score), 3)
        reward += bias_score
        breakdown["bias"] = bias_score
    else:
        breakdown["bias"] = 0.0

    # ── Alignment Check (max 0.25) ──────────────────────────────────────────
    align_correct = (action.alignment_violated == ground_truth["alignment_violated"])
    if align_correct:
        exp_score = llm_grade_explanation(
            action.alignment_explanation,
            ground_truth["alignment_reason"],
            "platform rule violation"
        )
        align_score = round(0.15 + (0.10 * exp_score), 3)
        reward += align_score
        breakdown["alignment"] = align_score
    else:
        breakdown["alignment"] = 0.0

    # ── Memory Check (max 0.15) ─────────────────────────────────────────────
    mem_correct = (action.memory_consistent == ground_truth["memory_consistent"])
    if mem_correct:
        exp_score = llm_grade_explanation(
            action.memory_explanation,
            ground_truth["memory_reason"],
            "author history and memory consistency"
        )
        mem_score = round(0.08 + (0.07 * exp_score), 3)
        reward += mem_score
        breakdown["memory"] = mem_score
    else:
        breakdown["memory"] = 0.0

    # ── Verdict Check (max 0.10) ────────────────────────────────────────────
    if action.overall_verdict == ground_truth["verdict"]:
        reward += 0.10
        breakdown["verdict"] = 0.10
    elif (
        action.overall_verdict == "borderline"
        and ground_truth["verdict"] in ["safe", "remove"]
    ):
        reward += 0.03
        breakdown["verdict"] = 0.03
    else:
        breakdown["verdict"] = 0.0

    # ── Overconfidence Penalty ──────────────────────────────────────────────
    if action.confidence > 0.85 and reward < 0.40:
        reward = max(0.0, reward - 0.10)
        breakdown["overconfidence_penalty"] = -0.10
    else:
        breakdown["overconfidence_penalty"] = 0.0

    reward = round(min(max(reward, 0.0), 1.0), 3)
    breakdown["total"] = reward

    return {"reward": reward, "breakdown": breakdown}