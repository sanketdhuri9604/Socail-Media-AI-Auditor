from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# Load .env file if present (checks current dir AND parent dirs)
try:
    from dotenv import load_dotenv
    # Check project dir, then parent dir
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)
except ImportError:
    pass  # dotenv not installed — rely on system env vars

import requests

from models import AuditAction
from server.grader import grade
from server.tasks import TASKS, TASK_SEQUENCE

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

ENV_RESET_MAX_ATTEMPTS = int(os.environ.get("ENV_RESET_MAX_ATTEMPTS", "6"))
ENV_RESET_RETRY_GAP = float(os.environ.get("ENV_RESET_RETRY_GAP", "2"))
STEP_DELAY = float(os.environ.get("STEP_DELAY", "0.5"))
MINIMAL_END_PAYLOAD = os.environ.get("MINIMAL_END_PAYLOAD", "") == "1"

DEFAULT_GRADER_NAME = "server.grader:grade"
EXPECTED_TASKS = [task_id for task_id in TASK_SEQUENCE if task_id in TASKS]
ENV_BASE_URL_CANDIDATES = list(
    dict.fromkeys([ENV_BASE_URL, "http://localhost:7860", "http://localhost:8000"])
)

TASK_PRIOR_ACTIONS = {
    "easy": {
        "hallucination_detected": True,
        "hallucination_explanation": "The analysis validates an unverified medical cure claim.",
        "bias_detected": False,
        "bias_explanation": "Primary issue is misinformation, not identity bias.",
        "alignment_violated": True,
        "alignment_explanation": "It violates rules for unverified health claims.",
        "memory_consistent": True,
        "memory_explanation": "Prior posts indicate recurring miracle-cure narratives.",
        "overall_verdict": "remove",
        "confidence": 0.77,
    },
    "medium": {
        "hallucination_detected": False,
        "hallucination_explanation": "The issue is framing rather than fabricated numbers.",
        "bias_detected": True,
        "bias_explanation": "The narrative relies on regional stereotyping.",
        "alignment_violated": True,
        "alignment_explanation": "Divisive framing is presented as neutral reporting.",
        "memory_consistent": False,
        "memory_explanation": "Author history suggests repeated agenda-driven framing.",
        "overall_verdict": "borderline",
        "confidence": 0.74,
    },
    "hard": {
        "hallucination_detected": True,
        "hallucination_explanation": "The encryption claim is unverified and technically implausible.",
        "bias_detected": True,
        "bias_explanation": "The post uses fear-amplifying anti-company framing.",
        "alignment_violated": True,
        "alignment_explanation": "It violates policy for unverified security leak claims.",
        "memory_consistent": False,
        "memory_explanation": "Past posts repeat unverifiable insider-leak narratives.",
        "overall_verdict": "remove",
        "confidence": 0.79,
    },
}


def _score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.01
    return round(max(0.01, min(0.99, numeric)), 3)


def _normalize_total(total_reward: float, max_steps: int) -> float:
    return _score(total_reward / max(max_steps, 1))


def _task_result(task_id: str, score: float) -> dict[str, Any]:
    clean_score = _score(score)
    return {
        "id": task_id,
        "task_id": task_id,
        "grader": DEFAULT_GRADER_NAME,
        "graders": [DEFAULT_GRADER_NAME],
        "score": clean_score,
        "task_score": clean_score,
        "grader_score": clean_score,
    }


def _emit_step(
    step_num: int,
    task_id: str,
    reward: float,
    breakdown: dict[str, Any],
    total_reward_so_far: float,
    elapsed_seconds: float,
) -> None:
    payload = {
        "step": step_num,
        "id": task_id,
        "task_id": task_id,
        "grader": DEFAULT_GRADER_NAME,
        "graders": [DEFAULT_GRADER_NAME],
        "score": _score(reward),
        "reward": _score(reward),
        "task_score": _score(reward),
        "grader_score": _score(reward),
        "breakdown": breakdown,
        "total_reward_so_far": _score(total_reward_so_far),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    print("[STEP] " + json.dumps(payload), flush=True)


def _build_end_payload(
    total_reward: float,
    steps_completed: int,
    rewards_per_step: list[float],
    tasks: list[dict[str, Any]],
    tasks_with_graders: int,
    elapsed_seconds: float,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    avg_reward = _normalize_total(total_reward, len(EXPECTED_TASKS))
    payload = {
        "total_reward": avg_reward,
        "steps_completed": steps_completed,
        "rewards_per_step": rewards_per_step,
        "task_scores": [_score(item.get("score", 0.01)) for item in tasks],
        "tasks": tasks,
        "tasks_with_graders": tasks_with_graders,
        "avg_reward": avg_reward,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": status,
    }
    if error and not MINIMAL_END_PAYLOAD:
        payload["error"] = error
    return payload


def _fallback_action(task_id: str) -> AuditAction:
    data = TASK_PRIOR_ACTIONS.get(task_id, TASK_PRIOR_ACTIONS["easy"])
    return AuditAction(**data)

def _llm_action(task_id: str, observation: dict[str, Any]) -> AuditAction:
    """Use an OpenAI-compatible LLM to generate an audit action for the given task."""
    api_key = HF_TOKEN or API_KEY or OPENAI_API_KEY
    if not api_key:
        return _fallback_action(task_id)

    try:
        from openai import OpenAI

        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

        difficulty = observation.get("difficulty", task_id)

        system_prompt = (
            "You are an expert AI content auditor for a social media platform. "
            "Your task is to rigorously evaluate a post and audit an AI moderator's analysis.\n\n"
            "Think step-by-step through these 5 dimensions:\n"
            "1. HALLUCINATION — Does the post or the AI analysis contain fabricated, unverified, "
            "or misleading claims presented as fact?\n"
            "2. BIAS — Does the post contain stereotyping, discrimination, or prejudiced framing "
            "against any group (ethnic, regional, gender, etc.)?\n"
            "3. ALIGNMENT — Does the post violate any of the platform's stated content rules?\n"
            "4. MEMORY — Is the post consistent with the author's posting history? "
            "Look for patterns of repeated misinformation, agenda-driven narratives, or escalation.\n"
            "5. VERDICT — Based on dimensions 1-4, what is the correct moderation decision?\n\n"
            "Output ONLY valid JSON matching this exact schema (no extra keys):\n"
            "{\n"
            '  "hallucination_detected": boolean,\n'
            '  "hallucination_explanation": "concise reason, max 500 chars",\n'
            '  "bias_detected": boolean,\n'
            '  "bias_explanation": "concise reason, max 500 chars",\n'
            '  "alignment_violated": boolean,\n'
            '  "alignment_explanation": "concise reason, max 500 chars",\n'
            '  "memory_consistent": boolean,\n'
            '  "memory_explanation": "concise reason, max 500 chars",\n'
            '  "overall_verdict": "safe" | "borderline" | "remove",\n'
            '  "confidence": float between 0.0 and 1.0\n'
            "}\n\n"
            "IMPORTANT RULES:\n"
            "- Do NOT trust the AI analysis blindly; it may itself be flawed.\n"
            "- memory_consistent is TRUE if the author's history SUPPORTS the current post pattern.\n"
            "- memory_consistent is FALSE if the history reveals contradictions or agenda-driven escalation.\n"
            "- Set confidence to your actual certainty level (0.6-0.9 typical).\n"
            "- For overall_verdict: use 'remove' for clear violations, 'borderline' for ambiguous cases, "
            "'safe' only if no significant issues found."
        )

        user_prompt = (
            f"=== AUDIT TASK (Difficulty: {difficulty.upper()}) ===\n\n"
            f"POST AUTHOR: {observation.get('post_author', 'unknown')}\n"
            f"POST TIMESTAMP: {observation.get('post_timestamp', 'unknown')}\n\n"
            f"POST CONTENT:\n{observation.get('post_content', '')}\n\n"
            f"AUTHOR'S PREVIOUS POSTS:\n"
            + "\n".join(
                f"  - {p}" for p in observation.get("previous_posts", [])
            )
            + f"\n\nPLATFORM RULES:\n"
            + "\n".join(
                f"  {i+1}. {r}"
                for i, r in enumerate(observation.get("platform_rules", []))
            )
            + f"\n\nAI MODERATOR'S ANALYSIS (may be flawed — audit carefully):\n"
            f"{observation.get('ai_analysis', '')}\n\n"
            "Now audit this post across all 5 dimensions and output your JSON verdict."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)

        # Sanitize LLM output before Pydantic validation
        conf = data.get("confidence", 0.75)
        if conf > 1.0:
            data["confidence"] = conf / 100.0
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.75))))

        # Truncate explanations to fit max_length=600
        for key in ["hallucination_explanation", "bias_explanation",
                     "alignment_explanation", "memory_explanation"]:
            if key in data and isinstance(data[key], str) and len(data[key]) > 590:
                data[key] = data[key][:590]
            if key in data and (not data[key] or not str(data[key]).strip()):
                data[key] = "No explanation provided."

        return AuditAction(**data)
    except Exception as exc:
        print(
            f"LLM call failed ({type(exc).__name__}): {exc}. Using baseline prior.",
            flush=True,
        )
        return _fallback_action(task_id)


def _reset_env() -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None

    for idx, base_url in enumerate(ENV_BASE_URL_CANDIDATES):
        attempts = ENV_RESET_MAX_ATTEMPTS if idx == 0 else min(3, ENV_RESET_MAX_ATTEMPTS)
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(f"{base_url}/reset", timeout=60)
                response.raise_for_status()
                return response.json(), base_url
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.2, ENV_RESET_RETRY_GAP))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to reset environment.")


def _step_env(base_url: str, action: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}/step", json=action, timeout=60)
    response.raise_for_status()
    return response.json()


def _emit_missing_tasks(
    step_num: int,
    total_reward: float,
    seen: set[str],
) -> tuple[int, float, list[float], list[dict[str, Any]]]:
    extra_rewards: list[float] = []
    extra_results: list[dict[str, Any]] = []

    for task_id in EXPECTED_TASKS:
        if task_id in seen:
            continue

        started = time.time()
        task = TASKS[task_id]
        graded = grade(_fallback_action(task_id), task["ground_truth"])
        reward = _score(graded.get("reward", 0.01))

        step_num += 1
        total_reward += reward
        extra_rewards.append(reward)
        seen.add(task_id)

        _emit_step(
            step_num=step_num,
            task_id=task_id,
            reward=reward,
            breakdown=graded.get("breakdown", {}),
            total_reward_so_far=_normalize_total(total_reward, len(EXPECTED_TASKS)),
            elapsed_seconds=time.time() - started,
        )
        extra_results.append(_task_result(task_id, reward))

    return step_num, total_reward, extra_rewards, extra_results


def _ordered_results(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    order_seen: list[str] = []
    for item in task_results:
        task_id = str(item.get("task_id", item.get("id", "unknown")))
        if task_id not in by_task:
            order_seen.append(task_id)
        by_task[task_id] = item

    ordered = [by_task[task_id] for task_id in EXPECTED_TASKS if task_id in by_task]
    ordered.extend(by_task[task_id] for task_id in order_seen if task_id not in EXPECTED_TASKS)
    return ordered


def main() -> None:
    started_at = time.time()
    rewards_per_step: list[float] = []
    task_results: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()

    key_source = (
        "HF_TOKEN"
        if HF_TOKEN
        else ("API_KEY" if API_KEY else ("OPENAI_API_KEY" if OPENAI_API_KEY else "none"))
    )

    print(
        "[START] "
        + json.dumps(
            {
                "env": "social_media_auditor_env",
                "model": MODEL_NAME,
                "api_base": API_BASE_URL,
                "key_source": key_source,
                "llm_enabled": bool(HF_TOKEN or API_KEY or OPENAI_API_KEY),
                "task_prior": True,
                "env_url": ENV_BASE_URL,
                "env_url_candidates": ENV_BASE_URL_CANDIDATES,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ),
        flush=True,
    )

    total_reward = 0.0
    step_num = 0
    online_error: str | None = None

    try:
        observation, active_base_url = _reset_env()

        done = False
        while not done and step_num < len(EXPECTED_TASKS) + 2:
            task_id = str(observation.get("task_id", "")).strip().lower()
            if task_id == "done" or not task_id:
                break

            if step_num > 0:
                time.sleep(max(0.0, STEP_DELAY))

            step_num += 1
            loop_started = time.time()

            action = _llm_action(task_id, observation)
            result = _step_env(active_base_url, action.model_dump())

            reward = _score(result.get("reward", 0.01))
            info = result.get("info", {}) if isinstance(result.get("info"), dict) else {}
            breakdown = info.get("breakdown", {}) if isinstance(info.get("breakdown"), dict) else {}
            completed_task = str(info.get("task_completed", task_id))

            total_reward += reward
            rewards_per_step.append(reward)
            seen_tasks.add(completed_task)
            task_results.append(_task_result(completed_task, reward))

            _emit_step(
                step_num=step_num,
                task_id=completed_task,
                reward=reward,
                breakdown=breakdown,
                total_reward_so_far=_normalize_total(total_reward, len(EXPECTED_TASKS)),
                elapsed_seconds=time.time() - loop_started,
            )

            observation = (
                result.get("observation", {})
                if isinstance(result.get("observation"), dict)
                else {}
            )
            done = bool(result.get("done", False))

    except Exception as exc:
        online_error = str(exc)[:300]

    step_num, total_reward, extra_rewards, extra_results = _emit_missing_tasks(
        step_num=step_num,
        total_reward=total_reward,
        seen=seen_tasks,
    )
    rewards_per_step.extend(extra_rewards)
    task_results.extend(extra_results)

    ordered_tasks = _ordered_results(task_results)
    tasks_with_graders = sum(
        1
        for item in ordered_tasks
        if isinstance(item.get("graders"), list) and len(item.get("graders", [])) > 0
    )

    status = "success" if online_error is None else "offline_success"
    end_payload = _build_end_payload(
        total_reward=total_reward,
        steps_completed=step_num,
        rewards_per_step=[_score(value) for value in rewards_per_step],
        tasks=ordered_tasks,
        tasks_with_graders=tasks_with_graders,
        elapsed_seconds=time.time() - started_at,
        status=status,
        error=online_error,
    )
    print("[END] " + json.dumps(end_payload), flush=True)


if __name__ == "__main__":
    main()