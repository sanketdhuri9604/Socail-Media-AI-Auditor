"""
environment.py — Social Media AI Auditor

Enhancements:
  - Task order is randomized on every reset so episodes feel fresh.
  - Per-dimension cumulative scores tracked in state for richer analytics.
"""

import uuid
import random
from datetime import datetime
from openenv_core.env_server import Environment
from models import AuditAction, AuditObservation
from server.tasks import TASKS
from server.grader import grade

TASK_ORDER_DEFAULT = ["easy", "medium", "hard", "expert", "bonus"]
MAX_STEPS = len(TASK_ORDER_DEFAULT)

_STATE = {
    "episode_id": None,
    "step_count": 0,
    "current_task_key": "easy",
    "task_order": list(TASK_ORDER_DEFAULT),
    "task_index": 0,
    "total_reward": 0.0,
    "history": [],
    "done": False,
    "started_at": None,
    "dimension_totals": {
        "hallucination": 0.0,
        "bias": 0.0,
        "alignment": 0.0,
        "memory": 0.0,
        "verdict": 0.0,
    },
}


class SocialMediaAuditorEnvironment(Environment):

    def reset(self) -> AuditObservation:
        shuffled = list(TASK_ORDER_DEFAULT)
        random.shuffle(shuffled)

        _STATE.update({
            "episode_id": str(uuid.uuid4()),
            "step_count": 0,
            "task_index": 0,
            "task_order": shuffled,
            "current_task_key": shuffled[0],
            "total_reward": 0.0,
            "history": [],
            "done": False,
            "started_at": datetime.utcnow().isoformat(),
            "dimension_totals": {
                "hallucination": 0.0,
                "bias": 0.0,
                "alignment": 0.0,
                "memory": 0.0,
                "verdict": 0.0,
            },
        })
        return self._build_observation()

    def step(self, action: AuditAction) -> tuple[AuditObservation, float, bool, dict]:
        if _STATE["done"]:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")

        task_key = _STATE["current_task_key"]
        task = TASKS[task_key]

        result = grade(action, task["ground_truth"])
        reward = result["reward"]
        breakdown = result["breakdown"]

        _STATE["total_reward"] += reward
        _STATE["step_count"] += 1
        _STATE["history"].append({
            "task": task_key,
            "reward": reward,
            "breakdown": breakdown,
        })

        for dim in ["hallucination", "bias", "alignment", "memory", "verdict"]:
            _STATE["dimension_totals"][dim] = round(
                _STATE["dimension_totals"][dim] + breakdown.get(dim, 0.0), 3
            )

        info = {
            "task_completed": task_key,
            "breakdown": breakdown,
            "total_reward_so_far": round(_STATE["total_reward"], 3),
        }

        _STATE["task_index"] += 1

        if _STATE["task_index"] >= len(_STATE["task_order"]):
            _STATE["done"] = True
            obs = self._build_observation(final=True)
            return obs, reward, True, info
        else:
            _STATE["current_task_key"] = _STATE["task_order"][_STATE["task_index"]]
            obs = self._build_observation()
            return obs, reward, False, info

    @property
    def state(self) -> dict:
        steps = max(_STATE["step_count"], 1)
        return {
            "episode_id": _STATE["episode_id"],
            "step_count": _STATE["step_count"],
            "task_index": _STATE["task_index"],
            "current_task": _STATE["current_task_key"],
            "task_order": _STATE["task_order"],
            "total_reward": round(_STATE["total_reward"], 3),
            "done": _STATE["done"],
            "history": _STATE["history"],
            "started_at": _STATE["started_at"],
            "dimension_totals": _STATE["dimension_totals"],
            "dimension_averages": {
                k: round(v / steps, 3)
                for k, v in _STATE["dimension_totals"].items()
            },
        }

    def _build_observation(self, final: bool = False) -> AuditObservation:
        if final:
            return AuditObservation(
                post_content="Episode complete.",
                post_author="",
                post_timestamp="",
                previous_posts=[],
                ai_analysis=f"Total reward: {round(_STATE['total_reward'], 3)} / {MAX_STEPS}.0",
                platform_rules=[],
                task_id="done",
                difficulty="done",
                step_number=_STATE["step_count"],
                max_steps=MAX_STEPS,
                reward=_STATE["total_reward"],
                done=True,
            )

        task = TASKS[_STATE["current_task_key"]]
        return AuditObservation(
            post_content=task["post_content"],
            post_author=task["post_author"],
            post_timestamp=task["post_timestamp"],
            previous_posts=task["previous_posts"],
            ai_analysis=task["ai_analysis"],
            platform_rules=task["platform_rules"],
            task_id=_STATE["current_task_key"],
            difficulty=_STATE["current_task_key"],
            step_number=_STATE["step_count"],
            max_steps=MAX_STEPS,
            reward=0.0,
            done=False,
        )