from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_VERDICTS = {"safe", "borderline", "remove"}


class AuditAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hallucination_detected: bool
    hallucination_explanation: str = Field(min_length=1, max_length=600)
    bias_detected: bool
    bias_explanation: str = Field(min_length=1, max_length=600)
    alignment_violated: bool
    alignment_explanation: str = Field(min_length=1, max_length=600)
    memory_consistent: bool
    memory_explanation: str = Field(min_length=1, max_length=600)
    overall_verdict: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("overall_verdict")
    @classmethod
    def validate_verdict(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_VERDICTS:
            raise ValueError("overall_verdict must be one of: safe, borderline, remove")
        return normalized


class AuditObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_content: str
    post_author: str
    post_timestamp: str
    previous_posts: list[str]
    ai_analysis: str
    platform_rules: list[str]
    task_id: str
    difficulty: str
    step_number: int
    max_steps: int
    reward: float = Field(default=0.01, gt=0.0, lt=1.0)
    done: bool = False