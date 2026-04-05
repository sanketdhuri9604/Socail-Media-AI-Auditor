from pydantic import BaseModel
from typing import Optional


class AuditAction(BaseModel):
    """What the agent submits after analyzing the post + AI analysis."""

    # Hallucination check
    hallucination_detected: bool
    hallucination_explanation: str  # "The post claims X but actually Y"

    # Bias check
    bias_detected: bool
    bias_explanation: str  # "This post is biased against group X because..."

    # Alignment check (did AI follow platform rules)
    alignment_violated: bool
    alignment_explanation: str  # "Platform rule Z was violated because..."

    # Memory check
    memory_consistent: bool
    memory_explanation: str  # "Previous post said X, this contradicts it"

    # Final verdict
    overall_verdict: str  # "safe" | "borderline" | "remove"
    confidence: float  # 0.0 to 1.0


class AuditObservation(BaseModel):
    """What the agent sees — the post + AI analysis to audit."""

    # The social media post
    post_content: str
    post_author: str
    post_timestamp: str

    # Previous posts from same author (for memory check)
    previous_posts: list[str]

    # AI-generated analysis of this post (may contain errors)
    ai_analysis: str

    # Platform rules (for alignment check)
    platform_rules: list[str]

    # Task metadata
    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    step_number: int
    max_steps: int