"""Local pre-submission validator for Social Media Auditor OpenEnv project."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from models import AuditAction, AuditObservation
from server.grader import grade
from server.tasks import TASKS

ROOT = Path(__file__).resolve().parent


def _check(condition: bool, label: str, details: str = "") -> tuple[bool, str, str]:
    return condition, label, details


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run() -> int:
    checks: list[tuple[bool, str, str]] = []

    required_files = [
        ROOT / "inference.py",
        ROOT / "openenv.yaml",
        ROOT / "Dockerfile",
        ROOT / "README.md",
        ROOT / "models.py",
        ROOT / "server" / "environment.py",
        ROOT / "server" / "grader.py",
        ROOT / "server" / "tasks.py",
    ]
    for f in required_files:
        checks.append(_check(f.exists(), f"file_exists:{f.name}", str(f)))

    checks.append(
        _check(len(TASKS) >= 3, "min_tasks", f"found={len(TASKS)}")
    )

    yaml_text = _read(ROOT / "openenv.yaml")
    task_ids = re.findall(r"^\s*-\s*id\s*:\s*([a-zA-Z0-9_\-]+)", yaml_text, flags=re.MULTILINE)
    checks.append(_check(len(task_ids) >= 3, "openenv_yaml_tasks", f"found={len(task_ids)}"))

    inference_text = _read(ROOT / "inference.py")
    for marker in ["[START]", "[STEP]", "[END]"]:
        checks.append(_check(marker in inference_text, f"inference_marker:{marker}"))

    for var in ["API_BASE_URL", "MODEL_NAME", "API_KEY", "HF_TOKEN"]:
        checks.append(_check(var in inference_text, f"inference_env_var:{var}"))
    checks.append(_check("OPENAI_API_KEY" in inference_text, "inference_env_var:OPENAI_API_KEY"))

    checks.append(_check(hasattr(AuditAction, "model_fields"), "typed_model:AuditAction"))
    checks.append(_check(hasattr(AuditObservation, "model_fields"), "typed_model:AuditObservation"))

    perfect_scores = {}
    for key, task in TASKS.items():
        gt = task["ground_truth"]
        action = AuditAction(
            hallucination_detected=gt["hallucination"],
            hallucination_explanation=gt["hallucination_reason"],
            bias_detected=gt["bias"],
            bias_explanation=gt["bias_reason"],
            alignment_violated=gt["alignment_violated"],
            alignment_explanation=gt["alignment_reason"],
            memory_consistent=gt["memory_consistent"],
            memory_explanation=gt["memory_reason"],
            overall_verdict=gt["verdict"],
            confidence=0.7,
        )
        perfect_scores[key] = grade(action, gt)["reward"]

    in_range = all(0.0 <= v <= 1.0 for v in perfect_scores.values())
    checks.append(_check(in_range, "grader_range_perfect", json.dumps(perfect_scores)))

    fixed_action = AuditAction(
        hallucination_detected=True,
        hallucination_explanation="x",
        bias_detected=True,
        bias_explanation="x",
        alignment_violated=True,
        alignment_explanation="x",
        memory_consistent=False,
        memory_explanation="x",
        overall_verdict="remove",
        confidence=0.95,
    )
    fixed_scores = {k: grade(fixed_action, v["ground_truth"])["reward"] for k, v in TASKS.items()}
    checks.append(
        _check(len(set(fixed_scores.values())) > 1, "grader_non_constant", json.dumps(fixed_scores))
    )

    passed = [c for c in checks if c[0]]
    failed = [c for c in checks if not c[0]]

    print("=== PRE-SUBMISSION VALIDATION ===")
    for ok, label, details in checks:
        status = "PASS" if ok else "FAIL"
        if details:
            print(f"[{status}] {label} :: {details}")
        else:
            print(f"[{status}] {label}")

    print(f"\nSummary: {len(passed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
