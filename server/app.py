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
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Social Media Auditor Environment</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-color: #030712;
      --card-bg: rgba(17, 24, 39, 0.7);
      --primary: #8b5cf6;
      --primary-hover: #7c3aed;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: 'Outfit', sans-serif; min-height: 100vh;
      background: radial-gradient(circle at top center, #1e1b4b 0%, var(--bg-color) 60%);
      color: var(--text-main); display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
    }
    .container { max-width: 1000px; width: 100%; display: flex; flex-direction: column; gap: 30px; }
    .header { text-align: center; margin-bottom: 20px; animation: fadeInDown 0.8s ease-out; }
    .header h1 {
      font-size: 3rem; font-weight: 800; background: linear-gradient(to right, #a855f7, #3b82f6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; letter-spacing: -1px;
    }
    .header p { font-size: 1.1rem; color: var(--text-muted); font-weight: 300; }
    .panel {
      background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 20px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
      animation: fadeInUp 0.8s ease-out;
    }
    .controls { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
    .status-text { font-size: 1.1rem; color: var(--text-muted); display: flex; align-items: center; gap: 10px; }
    .pulse { width: 10px; height: 10px; border-radius: 50%; background-color: var(--text-muted); transition: background-color 0.3s ease; }
    .pulse.active { background-color: var(--primary); box-shadow: 0 0 12px var(--primary); animation: pulsing 1.5s infinite; }
    .pulse.success { background-color: var(--success); box-shadow: 0 0 12px var(--success); }
    .pulse.error { background-color: var(--danger); box-shadow: 0 0 12px var(--danger); }
    button { 
      background: linear-gradient(135deg, var(--primary), var(--primary-hover)); color: white; border: none; border-radius: 12px; padding: 14px 28px; 
      font-size: 1.1rem; font-family: inherit; font-weight: 600; cursor: pointer; transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3); display: inline-flex; align-items: center; gap: 10px;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5); }
    button:disabled { background: #334155; color: #94a3b8; cursor: not-allowed; box-shadow: none; transform: none; }
    .results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 20px; opacity: 0; transform: translateY(20px); transition: all 0.5s ease; }
    .results-grid.visible { opacity: 1; transform: translateY(0); }
    .task-card {
      background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; transition: transform 0.3s ease, border-color 0.3s ease; position: relative; overflow: hidden;
    }
    .task-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, var(--primary), transparent); opacity: 0; transition: opacity 0.3s ease; }
    .task-card:hover { transform: translateY(-5px); border-color: rgba(139, 92, 246, 0.3); }
    .task-card:hover::before { opacity: 1; }
    .task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
    .task-title { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; text-transform: capitalize; }
    .score-badge { padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 800; }
    .score-high { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3);}
    .score-med { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3);}
    .score-low { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3);}
    .breakdown { font-size: 0.9rem; margin-top: 15px; border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 15px; }
    .breakdown-item { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .breakdown-label { color: var(--text-muted); }
    .breakdown-value { font-weight: 600; }
    .summary-box { margin-top: 30px; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 12px; padding: 20px; display: flex; justify-content: space-around; opacity: 0; transition: opacity 0.5s 0.2s ease; }
    .summary-box.visible { opacity: 1; }
    .summary-stat { text-align: center; }
    .summary-value { font-size: 2rem; font-weight: 800; color: var(--primary); }
    .summary-label { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
    .loader { display: none; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 1s linear infinite; }
    pre { background: #020617; color: #a5b4fc; border-radius: 12px; padding: 20px; overflow: auto; display: none; margin-top: 20px; }
    pre.visible { display: block; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulsing { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.6; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }
    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-30px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Auditor Analytics</h1>
      <p>Autonomous Agent Moderation Environment</p>
    </div>
    <div class="panel">
      <div class="controls">
        <div class="status-text">
          <div id="status-pulse" class="pulse"></div>
          <span id="status-msg">System Ready</span>
        </div>
        <button id="run">
          <span class="loader" id="spinner"></span>
          <span id="btn-text">Execute Evaluation</span>
          <svg id="btn-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        </button>
      </div>
      <div id="results" class="results-grid"></div>
      <div id="summary" class="summary-box">
        <div class="summary-stat">
          <div id="sum-steps" class="summary-value">-</div>
          <div class="summary-label">Steps</div>
        </div>
        <div class="summary-stat">
          <div id="sum-avg" class="summary-value">-</div>
          <div class="summary-label">Avg Reward</div>
        </div>
        <div class="summary-stat">
          <div id="sum-status" class="summary-value" style="color: var(--success); font-size: 1.5rem; margin-top: 5px;">-</div>
          <div class="summary-label">Status</div>
        </div>
      </div>
      <pre id="out"></pre>
    </div>
  </div>
  <script>
    const runBtn = document.getElementById("run"), spinner = document.getElementById("spinner"), btnText = document.getElementById("btn-text"), btnIcon = document.getElementById("btn-icon"), out = document.getElementById("out"), statusMsg = document.getElementById("status-msg"), statusPulse = document.getElementById("status-pulse"), resultsGrid = document.getElementById("results"), summaryBox = document.getElementById("summary");
    function getScoreClass(score) { return score >= 0.7 ? 'score-high' : (score >= 0.4 ? 'score-med' : 'score-low'); }
    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true; spinner.style.display = "block"; btnText.textContent = "Processing..."; btnIcon.style.display = "none";
      out.classList.remove("visible"); resultsGrid.classList.remove("visible"); summaryBox.classList.remove("visible");
      resultsGrid.innerHTML = ""; statusMsg.textContent = "Running Environment Baseline Simulation..."; statusPulse.className = "pulse active";
      try {
        const res = await fetch("/run_full", { method: "POST" });
        const data = await res.json();
        statusMsg.textContent = "Evaluation Complete"; statusPulse.className = "pulse success";
        if (data.status === "success") {
          data.steps.forEach(step => {
            const card = document.createElement("div"); card.className = "task-card";
            let breakdownHtml = "";
            let overallScore = `<span class="score-badge ${getScoreClass(step.score)}">${step.score.toFixed(2)}</span>`;
            if (step.breakdown) {
              const items = Object.entries(step.breakdown).map(([k, v]) => `<div class="breakdown-item"><span class="breakdown-label">${k.replace(/_/g, ' ')}</span><span class="breakdown-value" style="color: ${v > 0.5 ? 'var(--success)' : (v > 0.2 ? 'var(--warning)' : 'var(--danger)')}">${Number(v).toFixed(2)}</span></div>`).join('');
              breakdownHtml = `<div class="breakdown">${items}</div>`;
            }
            card.innerHTML = `<div class="task-header"><div class="task-title">${step.task_id} Task</div>${overallScore}</div><div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 5px;">Grader Score: ${step.grader_score.toFixed(2)}</div>${breakdownHtml}`;
            resultsGrid.appendChild(card);
          });
          document.getElementById("sum-steps").textContent = data.summary.steps_completed;
          document.getElementById("sum-avg").textContent = data.summary.avg_reward.toFixed(2);
          document.getElementById("sum-status").textContent = "SUCCESS";
          requestAnimationFrame(() => { resultsGrid.classList.add("visible"); summaryBox.classList.add("visible"); });
        } else { out.textContent = JSON.stringify(data, null, 2); out.classList.add("visible"); }
      } catch (err) {
        statusMsg.textContent = "Error Occurred"; statusPulse.className = "pulse error"; out.textContent = String(err); out.classList.add("visible");
      } finally {
        runBtn.disabled = false; spinner.style.display = "none"; btnText.textContent = "Execute Evaluation"; btnIcon.style.display = "block";
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