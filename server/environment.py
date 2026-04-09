from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from openenv_core.env_server import Environment

from models import AuditAction, AuditObservation
from server.grader import grade
from server.tasks import TASKS, TASK_SEQUENCE


def _clamp_score(value: float) -> float:
    return round(max(0.001, min(0.999, float(value))), 3)


class SocialMediaAuditorEnvironment(Environment):
    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self.reset()

    def reset(self) -> AuditObservation:
        self._state = {
            "episode_id": str(uuid.uuid4()),
            "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "task_order": list(TASK_SEQUENCE),
            "task_index": 0,
            "step_count": 0,
            "total_reward_raw": 0.0,
            "done": False,
            "history": [],
        }
        return self._current_observation()

    def step(self, action: AuditAction) -> tuple[AuditObservation, float, bool, dict[str, Any]]:
        if self._state["done"]:
            raise RuntimeError("Episode is complete. Call /reset before /step.")

        task_id = self._state["task_order"][self._state["task_index"]]
        task = TASKS[task_id]

        graded = grade(action, task["ground_truth"])
        reward = _clamp_score(graded["reward"])
        breakdown = graded["breakdown"]

        self._state["step_count"] += 1
        self._state["total_reward_raw"] += reward
        self._state["history"].append(
            {
                "task_id": task_id,
                "reward": reward,
                "breakdown": breakdown,
            }
        )

        info = {
            "task_completed": task_id,
            "status": "ok",
            "breakdown": breakdown,
            "total_reward_so_far": self._average_reward(),
        }

        self._state["task_index"] += 1
        if self._state["task_index"] >= len(self._state["task_order"]):
            self._state["done"] = True
            obs = self._terminal_observation()
            obs = obs.model_copy(update={"reward": reward, "done": True})
            return obs, reward, True, info

        obs = self._current_observation().model_copy(update={"reward": reward, "done": False})
        return obs, reward, False, info

    @property
    def state(self) -> dict[str, Any]:
        return {
            "episode_id": self._state.get("episode_id"),
            "started_at": self._state.get("started_at"),
            "step_count": self._state.get("step_count", 0),
            "task_index": self._state.get("task_index", 0),
            "task_order": self._state.get("task_order", []),
            "done": self._state.get("done", False),
            "total_reward": self._average_reward(),
            "history": self._state.get("history", []),
        }

    def _average_reward(self) -> float:
        return _clamp_score(self._state["total_reward_raw"] / max(len(TASK_SEQUENCE), 1))

    def _current_observation(self) -> AuditObservation:
        task_id = self._state["task_order"][self._state["task_index"]]
        task = TASKS[task_id]
        return AuditObservation(
            post_content=task["post_content"],
            post_author=task["post_author"],
            post_timestamp=task["post_timestamp"],
            previous_posts=task["previous_posts"],
            ai_analysis=task["ai_analysis"],
            platform_rules=task["platform_rules"],
            task_id=task_id,
            difficulty=task["difficulty"],
            step_number=self._state["step_count"],
            max_steps=len(TASK_SEQUENCE),
            reward=0.001,
            done=False,
        )

    def _terminal_observation(self) -> AuditObservation:
        return AuditObservation(
            post_content="Episode complete.",
            post_author="",
            post_timestamp="",
            previous_posts=[],
            ai_analysis="All tasks were processed.",
            platform_rules=[],
            task_id="done",
            difficulty="done",
            step_number=self._state["step_count"],
            max_steps=len(TASK_SEQUENCE),
            reward=self._average_reward(),
            done=True,
        )