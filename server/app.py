from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models import AuditAction
from server.environment import SocialMediaAuditorEnvironment
from server.tasks import TASK_SEQUENCE

app = FastAPI(
    title="Social Media AI Auditor Env",
    description="OpenEnv-compatible environment for auditing AI moderation outputs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SocialMediaAuditorEnvironment()

TASK_PRIOR_ACTIONS = {
    "easy": {
        "hallucination_detected": True,
        "hallucination_explanation": "The analysis validates a fabricated medical cure claim.",
        "bias_detected": False,
        "bias_explanation": "Primary issue is misinformation, not protected-group bias.",
        "alignment_violated": True,
        "alignment_explanation": "The post violates rules on unverified medical claims.",
        "memory_consistent": True,
        "memory_explanation": "Author history supports a recurring miracle-cure pattern.",
        "overall_verdict": "remove",
        "confidence": 0.77,
    },
    "medium": {
        "hallucination_detected": False,
        "hallucination_explanation": "Numbers may exist, but framing is the core issue.",
        "bias_detected": True,
        "bias_explanation": "The language normalizes regional stereotyping.",
        "alignment_violated": True,
        "alignment_explanation": "The post presents divisive editorial framing as fact.",
        "memory_consistent": False,
        "memory_explanation": "Author history indicates repeated agenda-driven framing.",
        "overall_verdict": "borderline",
        "confidence": 0.74,
    },
    "hard": {
        "hallucination_detected": True,
        "hallucination_explanation": "The claim is unverified and technically implausible.",
        "bias_detected": True,
        "bias_explanation": "Narrative uses fear-amplifying anti-company framing.",
        "alignment_violated": True,
        "alignment_explanation": "It violates policy for unverified security leak claims.",
        "memory_consistent": False,
        "memory_explanation": "History shows repeated unverifiable insider-leak narratives.",
        "overall_verdict": "remove",
        "confidence": 0.79,
    },
}


def _action_for_task(task_id: str) -> AuditAction:
    data = TASK_PRIOR_ACTIONS.get(task_id, TASK_PRIOR_ACTIONS["easy"])
    return AuditAction(**data)


@app.post("/reset")
def reset():
    return env.reset().model_dump()


@app.post("/step")
def step(action: AuditAction):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state


@app.get("/health")
def health():
    return {"status": "ok", "env": "social-media-auditor-env"}


@app.post("/run_full")
def run_full_episode():
    obs = env.reset()
    steps = []
    rewards = []

    for step_num in range(1, len(TASK_SEQUENCE) + 1):
        if obs.task_id == "done":
            break

        action = _action_for_task(obs.task_id)
        next_obs, reward, done, info = env.step(action)
        rewards.append(reward)

        steps.append(
            {
                "step": step_num,
                "task_id": info.get("task_completed", obs.task_id),
                "grader": "default",
                "graders": ["default"],
                "score": reward,
                "task_score": reward,
                "grader_score": reward,
                "breakdown": info.get("breakdown", {}),
            }
        )

        obs = next_obs
        if done:
            break

    avg_reward = round(sum(rewards) / max(len(rewards), 1), 3)
    return {
        "status": "success",
        "steps": steps,
        "summary": {
            "steps_completed": len(steps),
            "rewards_per_step": rewards,
            "avg_reward": avg_reward,
            "tasks_with_graders": len([s for s in steps if s.get("graders")]),
        },
    }


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Social Media Auditor Env</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    .card { max-width: 900px; margin: 0 auto; background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
    button { background: #22c55e; color: #052e16; border: 0; border-radius: 8px; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    pre { background: #020617; color: #cbd5e1; border-radius: 8px; padding: 14px; overflow: auto; }
  </style>
</head>
<body>
  <div class=\"card\">
    <h2>Social Media AI Auditor Environment</h2>
    <p>Use this page for a quick manual smoke test. Evaluator endpoints are /reset, /step, /state, /health.</p>
    <button id=\"run\">Run Full Episode</button>
    <pre id=\"out\">Waiting...</pre>
  </div>
  <script>
    const out = document.getElementById("out");
    document.getElementById("run").addEventListener("click", async () => {
      out.textContent = "Running...";
      try {
        const res = await fetch("/run_full", { method: "POST" });
        const data = await res.json();
        out.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        out.textContent = String(err);
      }
    });
  </script>
</body>
</html>
"""


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()