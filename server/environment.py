import uuid
from datetime import datetime
from openenv_core.env_server import Environment
from models import AuditAction, AuditObservation
from server.tasks import TASKS
from server.grader import grade
# Global state store
_STATE = {
    "episode_id": None,
    "step_count": 0,
    "current_task_key": "easy",
    "task_order": ["easy", "medium", "hard"],
    "task_index": 0,
    "total_reward": 0.0,
    "history": [],
    "done": False,
    "started_at": None,
}
class SocialMediaAuditorEnvironment(Environment):
    def reset(self) -> AuditObservation:
        _STATE["episode_id"] = str(uuid.uuid4())
        _STATE["step_count"] = 0
        _STATE["task_index"] = 0
        _STATE["total_reward"] = 0.0
        _STATE["history"] = []
        _STATE["done"] = False
        _STATE["started_at"] = datetime.utcnow().isoformat()
        _STATE["current_task_key"] = _STATE["task_order"][0]
        return self._build_observation()
    def step(self, action: AuditAction) -> AuditObservation:
        if _STATE["done"]:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")
        task = TASKS[_STATE["current_task_key"]]
        result = grade(action, task["ground_truth"])
        reward = result["reward"]
        _STATE["total_reward"] += reward
        _STATE["step_count"] += 1
        _STATE["history"].append({
            "task": _STATE["current_task_key"],
            "reward": reward,
            "breakdown": result["breakdown"],
        })
        _STATE["task_index"] += 1
        if _STATE["task_index"] >= len(_STATE["task_order"]):
            _STATE["done"] = True
            return self._build_observation(final=True)
        else:
            _STATE["current_task_key"] = _STATE["task_order"][_STATE["task_index"]]
            return self._build_observation()
    @property
    def state(self) -> dict:
        return {
            "episode_id": _STATE["episode_id"],
            "step_count": _STATE["step_count"],
            "task_index": _STATE["task_index"],
            "current_task": _STATE["current_task_key"],
            "total_reward": round(_STATE["total_reward"], 3),
            "done": _STATE["done"],
            "history": _STATE["history"],
            "started_at": _STATE["started_at"],
        }
    def _build_observation(self, final: bool = False) -> AuditObservation:
        if final:
            return AuditObservation(
                post_content="Episode complete.",
                post_author="",
                post_timestamp="",
                previous_posts=[],
                ai_analysis=f"Total reward: {round(_STATE['total_reward'], 3)}",
                platform_rules=[],
                task_id="done",
                difficulty="done",
                step_number=_STATE["step_count"],
                max_steps=3,
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
            max_steps=3,
            reward=0.0,
            done=False,
        )
