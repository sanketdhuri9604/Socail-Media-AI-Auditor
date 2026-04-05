from pydantic import BaseModel
from typing import Optional
class AuditAction(BaseModel):
    hallucination_detected: bool
    hallucination_explanation: str
    bias_detected: bool
    bias_explanation: str
    alignment_violated: bool
    alignment_explanation: str
    memory_consistent: bool
    memory_explanation: str
    overall_verdict: str
    confidence: float
class AuditObservation(BaseModel):
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
    reward: float = 0.0
    done: bool = False
