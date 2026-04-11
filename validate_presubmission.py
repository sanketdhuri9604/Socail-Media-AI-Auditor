from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from models import AuditAction, AuditObservation
from server.grader import grade_detailed, grade
from server.tasks import TASKS

ROOT = Path(__file__).resolve().parent


def _check(condition: bool, label: str, details: str = "") -> tuple[bool, str, str]:
    return condition, label, details


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_inference_contract_check() -> tuple[bool, str]:
    env = os.environ.copy()
    env.update(
        {
            "API_KEY": "",
            "HF_TOKEN": "",
            "OPENAI_API_KEY": "",
            "API_BASE_URL": "https://api.groq.com/openai/v1",
            "MODEL_NAME": "llama-3.3-70b-versatile",
            "ENV_BASE_URL": "http://localhost:65535",
            "MINIMAL_END_PAYLOAD": "1",
        }
    )

    proc = subprocess.run(
        [sys.executable, str(ROOT / "inference.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if proc.returncode != 0:
        return False, f"inference_exit_code={proc.returncode}"

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    step_lines = [line for line in lines if line.startswith("[STEP] ")]
    end_lines = [line for line in lines if line.startswith("[END] ")]

    if len(step_lines) < 3:
        return False, f"step_markers_found={len(step_lines)}"
    if not end_lines:
        return False, "missing_end_marker"

    try:
        end_payload = json.loads(end_lines[-1].replace("[END] ", "", 1))
    except json.JSONDecodeError as exc:
        return False, f"end_json_decode_error={exc}"

    tasks = end_payload.get("tasks", [])
    scores = end_payload.get("task_scores", [])
    tasks_with_graders = end_payload.get("tasks_with_graders", 0)
    score_range_ok = all(0.0 < float(v) < 1.0 for v in scores)

    condition = (
        isinstance(tasks, list)
        and len(tasks) >= 3
        and isinstance(tasks_with_graders, int)
        and tasks_with_graders >= 3
        and score_range_ok
    )
    details = json.dumps(
        {
            "tasks": len(tasks),
            "tasks_with_graders": tasks_with_graders,
            "task_scores": scores,
            "status": end_payload.get("status"),
        }
    )
    return condition, details


def run() -> int:
    checks: list[tuple[bool, str, str]] = []

    required_files = [
        ROOT / "app.py",
        ROOT / "inference.py",
        ROOT / "openenv.yaml",
        ROOT / "Dockerfile",
        ROOT / "README.md",
        ROOT / "models.py",
        ROOT / "server" / "app.py",
        ROOT / "server" / "environment.py",
        ROOT / "server" / "grader.py",
        ROOT / "server" / "tasks.py",
    ]
    for file_path in required_files:
        checks.append(_check(file_path.exists(), f"file_exists:{file_path.name}", str(file_path)))

    checks.append(_check(len(TASKS) >= 3, "min_tasks", f"found={len(TASKS)}"))

    grader_fields_ok = all(
        isinstance(task.get("grader"), str)
        and isinstance(task.get("graders"), list)
        and len(task.get("graders", [])) > 0
        for task in TASKS.values()
    )
    checks.append(_check(grader_fields_ok, "tasks_have_graders"))

    yaml_text = _read(ROOT / "openenv.yaml")
    task_ids = re.findall(r"^\s*-\s*id\s*:\s*([a-zA-Z0-9_\-]+)", yaml_text, flags=re.MULTILINE)
    checks.append(_check(len(task_ids) >= 3, "openenv_yaml_tasks", f"found={len(task_ids)}"))

    inference_text = _read(ROOT / "inference.py")
    for marker in ["[START]", "[STEP]", "[END]"]:
        checks.append(_check(marker in inference_text, f"inference_marker:{marker}"))

    for env_var in ["API_BASE_URL", "MODEL_NAME", "API_KEY", "HF_TOKEN", "OPENAI_API_KEY"]:
        checks.append(_check(env_var in inference_text, f"inference_env_var:{env_var}"))

    checks.append(_check(hasattr(AuditAction, "model_fields"), "typed_model:AuditAction"))
    checks.append(_check(hasattr(AuditObservation, "model_fields"), "typed_model:AuditObservation"))

    perfect_scores = {}
    perfect_breakdowns = {}
    for task_id, task in TASKS.items():
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
        result = grade_detailed(action, gt)
        perfect_scores[task_id] = result["reward"]
        perfect_breakdowns[task_id] = result["breakdown"]

    checks.append(
        _check(
            all(0.0 < value < 1.0 for value in perfect_scores.values()),
            "grader_range_perfect",
            json.dumps(perfect_scores),
        )
    )

    breakdown_values: list[float] = []
    for breakdown in perfect_breakdowns.values():
        for value in breakdown.values():
            if isinstance(value, (int, float)):
                breakdown_values.append(float(value))
    checks.append(
        _check(
            all(0.0 < value < 1.0 for value in breakdown_values),
            "grader_breakdown_range_perfect",
            json.dumps(perfect_breakdowns),
        )
    )

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
    fixed_scores = {
        task_id: grade(fixed_action, task["ground_truth"])
        for task_id, task in TASKS.items()
    }
    checks.append(
        _check(
            len(set(fixed_scores.values())) > 1,
            "grader_non_constant",
            json.dumps(fixed_scores),
        )
    )

    inference_contract_ok, inference_contract_details = _run_inference_contract_check()
    checks.append(
        _check(inference_contract_ok, "inference_runtime_contract", inference_contract_details)
    )

    passed = [item for item in checks if item[0]]
    failed = [item for item in checks if not item[0]]

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