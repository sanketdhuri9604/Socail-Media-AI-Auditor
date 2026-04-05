import uuid
from datetime import datetime
from openenv_core.env_server import Environment
from models import AuditAction, AuditObservation
from server.tasks import TASKS
from server.grader import grade
class SocialMediaAuditorEnvironment(Environment):
    def __init__(self):
        self.episode_id = None
        self.step_count = 0
        self.current_task_key = "easy"
        self.task_order = ["easy", "medium", "hard"]
        self.task_index = 0
        self.total_reward = 0.0
        self.history = []
        self.done = False
        self.started_at = None
    def reset(self) -> AuditObservation:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.task_index = 0
        self.total_reward = 0.0
        self.history = []
        self.done = False
        self.started_at = datetime.utcnow().isoformat()
        self.current_task_key = self.task_order[self.task_index]
        return self._build_observation()
    def step(self, action: AuditAction) -> tuple:
        if self.done:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")
        task = TASKS[self.current_task_key]
        result = grade(action, task["ground_truth"])
        reward = result["reward"]
        self.total_reward += reward
        self.step_count += 1
        self.history.append({
            "task": self.current_task_key,
            "reward": reward,
            "breakdown": result["breakdown"],
        })
        self.task_index += 1
        if self.task_index >= len(self.task_order):
            self.done = True
            obs = self._build_observation(final=True)
        else:
            self.current_task_key = self.task_order[self.task_index]
            obs = self._build_observation()
        info = {
            "breakdown": result["breakdown"],
            "total_reward_so_far": round(self.total_reward, 3),
        }
        return obs, reward, self.done, info
    @property
    def state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_index": self.task_index,
            "current_task": self.current_task_key,
            "total_reward": round(self.total_reward, 3),
            "done": self.done,
            "history": self.history,
            "started_at": self.started_at,
        }
    def _build_observation(self, final: bool = False) -> AuditObservation:
        if final:
            return AuditObservation(
                post_content="Episode complete.",
                post_author="",
                post_timestamp="",
                previous_posts=[],
                ai_analysis=f"Total reward: {round(self.total_reward, 3)}",
                platform_rules=[],
                task_id="done",
                difficulty="done",
                step_number=self.step_count,
                max_steps=3,
                reward=self.total_reward,
                done=True,
            )
        task = TASKS[self.current_task_key]
        return AuditObservation(
            post_content=task["post_content"],
            post_author=task["post_author"],
            post_timestamp=task["post_timestamp"],
            previous_posts=task["previous_posts"],
            ai_analysis=task["ai_analysis"],
            platform_rules=task["platform_rules"],
            task_id=self.current_task_key,
            difficulty=self.current_task_key,
            step_number=self.step_count,
            max_steps=3,
            reward=0.0,
            done=False,
        )
