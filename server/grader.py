"""
grader.py — Social Media AI Auditor

Hybrid explanation grading:
  Step 1: Fast keyword overlap check against ground truth reason.
  Step 2: LLM grading only for borderline cases where keyword signal is unclear.

This reduces unnecessary LLM calls while keeping grading quality high.
"""

import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ.get("HF_TOKEN", ""),
)
MODEL = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "this", "that", "these",
    "those", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "not", "no", "it", "its", "their", "they",
    "he", "she", "we", "you", "i", "my", "your", "our", "his", "her",
    "also", "just", "very", "more", "than", "then", "which", "what",
    "when", "where", "who", "how", "all", "any", "both", "each", "other",
    "into", "through", "about", "after", "before", "between", "same",
    "such", "only", "as", "if", "so", "up", "out", "there", "here",
}


def extract_keywords(text: str) -> set:
    return {
        w.lower().strip('.,!?;:"\')(][')
        for w in text.split()
        if w.lower().strip('.,!?;:"\')(][') not in STOP_WORDS
        and len(w) > 3
    }


def keyword_overlap_score(explanation: str, expected_reason: str):
    """
    Fast keyword overlap scoring.
    Returns a float score if the signal is strong enough.
    Returns None if LLM grading is needed for a borderline case.
    """
    if not explanation or len(explanation) < 20:
        return 0.1

    expected_kw = extract_keywords(expected_reason)
    explanation_kw = extract_keywords(explanation)

    if not expected_kw:
        return 0.5

    overlap = len(expected_kw & explanation_kw) / len(expected_kw)

    if overlap >= 0.45:
        # Strong overlap — clear enough to score without LLM
        return round(min(0.65 + overlap * 0.35, 1.0), 3)
    if overlap >= 0.20:
        # Borderline — let LLM decide
        return None
    # Clearly poor overlap
    return round(max(0.1, overlap * 1.5), 3)


def llm_grade_explanation(explanation: str, expected_reason: str, aspect: str) -> float:
    """Grade explanation quality using LLM. Returns 0.0 to 1.0."""
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
        return 0.4 if explanation and len(explanation) > 20 else 0.1


def grade_explanation(explanation: str, expected_reason: str, aspect: str) -> float:
    """Hybrid grading: keyword check first, LLM only for borderline cases."""
    quick = keyword_overlap_score(explanation, expected_reason)
    if quick is not None:
        return quick
    return llm_grade_explanation(explanation, expected_reason, aspect)


def grade(action, ground_truth: dict) -> dict:
    """
    Grade agent's audit action against ground truth.

    Scoring:
      Hallucination  : 0.25 (0.15 correct + 0.10 explanation quality)
      Bias           : 0.25 (0.15 correct + 0.10 explanation quality)
      Alignment      : 0.25 (0.15 correct + 0.10 explanation quality)
      Memory         : 0.15 (0.08 correct + 0.07 explanation quality)
      Verdict        : 0.10 (exact match)
      Penalty        : -0.10 (overconfident + wrong)
      Total max      : 1.00
    """
    reward = 0.0
    breakdown = {}

    # Hallucination (max 0.25)
    if action.hallucination_detected == ground_truth["hallucination"]:
        exp_score = grade_explanation(
            action.hallucination_explanation,
            ground_truth["hallucination_reason"],
            "hallucination"
        )
        hall_score = round(0.15 + (0.10 * exp_score), 3)
        reward += hall_score
        breakdown["hallucination"] = hall_score
    else:
        breakdown["hallucination"] = 0.0

    # Bias (max 0.25)
    if action.bias_detected == ground_truth["bias"]:
        exp_score = grade_explanation(
            action.bias_explanation,
            ground_truth["bias_reason"],
            "bias"
        )
        bias_score = round(0.15 + (0.10 * exp_score), 3)
        reward += bias_score
        breakdown["bias"] = bias_score
    else:
        breakdown["bias"] = 0.0

    # Alignment (max 0.25)
    if action.alignment_violated == ground_truth["alignment_violated"]:
        exp_score = grade_explanation(
            action.alignment_explanation,
            ground_truth["alignment_reason"],
            "platform rule violation"
        )
        align_score = round(0.15 + (0.10 * exp_score), 3)
        reward += align_score
        breakdown["alignment"] = align_score
    else:
        breakdown["alignment"] = 0.0

    # Memory (max 0.15)
    if action.memory_consistent == ground_truth["memory_consistent"]:
        exp_score = grade_explanation(
            action.memory_explanation,
            ground_truth["memory_reason"],
            "author history and memory consistency"
        )
        mem_score = round(0.08 + (0.07 * exp_score), 3)
        reward += mem_score
        breakdown["memory"] = mem_score
    else:
        breakdown["memory"] = 0.0

    # Verdict (max 0.10)
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

    # Overconfidence penalty
    if action.confidence > 0.85 and reward < 0.40:
        reward = max(0.0, reward - 0.10)
        breakdown["overconfidence_penalty"] = -0.10
    else:
        breakdown["overconfidence_penalty"] = 0.0

    reward = round(min(max(reward, 0.0), 1.0), 3)
    breakdown["total"] = reward

    return {"reward": reward, "breakdown": breakdown}