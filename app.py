import sys
import os
import json
import uuid
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models import AuditAction, AuditObservation
from server.environment import SocialMediaAuditorEnvironment
from server.tasks import TASKS

app = FastAPI(
    title="Social Media AI Auditor Env",
    description="OpenEnv RL environment for auditing AI-generated social media analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SocialMediaAuditorEnvironment()

# ── OpenEnv Required Endpoints ───────────────────────────────────────────────

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: AuditAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/health")
def health():
    return {"status": "ok", "env": "social-media-auditor-env", "version": "1.0.0"}

# ── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    tasks_preview = []
    for tid, t in TASKS.items():
        tasks_preview.append({
            "id": tid,
            "content": t["post_content"][:120] + "...",
            "author": t["post_author"],
            "verdict": t["ground_truth"]["verdict"],
        })

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Social Media AI Auditor — OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f13; color: #e2e2e8; min-height: 100vh; }

  .hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 48px 32px 40px; text-align: center; border-bottom: 1px solid #2a2a3e; }
  .hero-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); color: #a5b4fc; font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 20px; margin-bottom: 20px; letter-spacing: 0.5px; }
  .hero h1 { font-size: 36px; font-weight: 700; background: linear-gradient(135deg, #e2e2e8, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; }
  .hero p { color: #9ca3af; font-size: 16px; max-width: 560px; margin: 0 auto 28px; line-height: 1.6; }
  .hero-stats { display: flex; justify-content: center; gap: 40px; }
  .stat { text-align: center; }
  .stat-num { font-size: 28px; font-weight: 700; color: #a5b4fc; }
  .stat-label { font-size: 12px; color: #6b7280; margin-top: 2px; }

  .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }

  .card { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 16px; padding: 24px; }
  .card-title { font-size: 13px; font-weight: 600; color: #6366f1; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }

  .dimensions { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .dim { background: #0f0f1a; border: 1px solid #2a2a3e; border-radius: 10px; padding: 14px; }
  .dim-icon { font-size: 20px; margin-bottom: 6px; }
  .dim-name { font-size: 13px; font-weight: 600; color: #e2e2e8; margin-bottom: 4px; }
  .dim-desc { font-size: 11px; color: #6b7280; line-height: 1.4; }

  .task-card { background: #0f0f1a; border: 1px solid #2a2a3e; border-radius: 10px; padding: 16px; margin-bottom: 12px; cursor: pointer; transition: border-color 0.2s; }
  .task-card:hover { border-color: #6366f1; }
  .task-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .task-badge { font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
  .badge-easy { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
  .badge-medium { background: rgba(234,179,8,0.15); color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
  .badge-hard { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
  .verdict-badge { font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px; }
  .verdict-remove { background: rgba(239,68,68,0.2); color: #f87171; }
  .verdict-borderline { background: rgba(234,179,8,0.2); color: #facc15; }
  .verdict-safe { background: rgba(34,197,94,0.2); color: #4ade80; }
  .task-author { font-size: 11px; color: #6366f1; margin-bottom: 6px; }
  .task-content { font-size: 12px; color: #9ca3af; line-height: 1.5; }

  .tester { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 16px; padding: 24px; margin-bottom: 32px; }
  .tester-title { font-size: 13px; font-weight: 600; color: #6366f1; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 20px; }

  .btn-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .btn { padding: 10px 20px; border-radius: 8px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: #6366f1; color: white; }
  .btn-primary:hover { background: #5558e8; }
  .btn-secondary { background: #2a2a3e; color: #e2e2e8; border: 1px solid #3a3a4e; }
  .btn-secondary:hover { background: #3a3a4e; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .response-box { background: #0a0a12; border: 1px solid #2a2a3e; border-radius: 10px; padding: 16px; font-family: 'Courier New', monospace; font-size: 12px; color: #a5b4fc; white-space: pre-wrap; min-height: 80px; max-height: 400px; overflow-y: auto; line-height: 1.6; }
  .response-box.loading { color: #6b7280; }
  .response-box.error { color: #f87171; }
  .response-box.success { color: #4ade80; }

  .score-display { display: flex; align-items: center; gap: 16px; background: #0f0f1a; border-radius: 10px; padding: 16px; margin-top: 12px; }
  .score-num { font-size: 32px; font-weight: 700; color: #a5b4fc; }
  .score-label { font-size: 12px; color: #6b7280; }
  .score-bar { flex: 1; height: 8px; background: #2a2a3e; border-radius: 4px; overflow: hidden; }
  .score-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #a5b4fc); border-radius: 4px; transition: width 0.5s ease; }

  .endpoints { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  @media (max-width: 600px) { .endpoints { grid-template-columns: 1fr; } }
  .endpoint { background: #0f0f1a; border: 1px solid #2a2a3e; border-radius: 10px; padding: 14px; }
  .ep-method { font-size: 10px; font-weight: 700; color: #4ade80; margin-bottom: 4px; }
  .ep-path { font-size: 14px; font-weight: 600; color: #e2e2e8; font-family: monospace; margin-bottom: 4px; }
  .ep-desc { font-size: 11px; color: #6b7280; }

  .reward-breakdown { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 12px; }
  @media (max-width: 600px) { .reward-breakdown { grid-template-columns: repeat(2, 1fr); } }
  .rb-item { background: #0f0f1a; border: 1px solid #2a2a3e; border-radius: 8px; padding: 10px; text-align: center; }
  .rb-val { font-size: 18px; font-weight: 700; color: #a5b4fc; }
  .rb-label { font-size: 10px; color: #6b7280; margin-top: 2px; }

  .status-dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  footer { text-align: center; padding: 24px; color: #4b5563; font-size: 12px; border-top: 1px solid #2a2a3e; }
</style>
</head>
<body>

<div class="hero">
  <div class="hero-badge"><span class="status-dot"></span> LIVE — OpenEnv Compatible</div>
  <h1>Social Media AI Auditor</h1>
  <p>An RL environment that trains agents to detect hallucinations, bias, alignment failures, and memory inconsistencies in AI-generated social media analysis.</p>
  <div class="hero-stats">
    <div class="stat"><div class="stat-num">4</div><div class="stat-label">Audit Dimensions</div></div>
    <div class="stat"><div class="stat-num">3</div><div class="stat-label">Task Levels</div></div>
    <div class="stat"><div class="stat-num">1.0</div><div class="stat-label">Max Reward</div></div>
  </div>
</div>

<div class="container">

  <!-- Live Tester -->
  <div class="tester">
    <div class="tester-title">🧪 Live Environment Tester</div>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="doReset()">POST /reset</button>
      <button class="btn btn-secondary" onclick="doState()">GET /state</button>
      <button class="btn btn-secondary" onclick="doHealth()">GET /health</button>
      <button class="btn btn-secondary" id="step-btn" onclick="doStep()" disabled>POST /step (auto-agent)</button>
    </div>
    <div class="response-box loading" id="response">Click a button to test the environment...</div>
    <div id="score-section" style="display:none">
      <div class="score-display">
        <div>
          <div class="score-num" id="score-num">0.00</div>
          <div class="score-label">Episode Reward</div>
        </div>
        <div class="score-bar"><div class="score-fill" id="score-fill" style="width:0%"></div></div>
        <div style="font-size:12px;color:#6b7280" id="score-steps">0 / 3 tasks</div>
      </div>
      <div class="reward-breakdown" id="breakdown"></div>
    </div>
  </div>

  <div class="grid">
    <!-- Dimensions -->
    <div class="card">
      <div class="card-title">🔍 Audit Dimensions</div>
      <div class="dimensions">
        <div class="dim">
          <div class="dim-icon">🤥</div>
          <div class="dim-name">Hallucination</div>
          <div class="dim-desc">Did the AI fabricate facts that don't exist?</div>
        </div>
        <div class="dim">
          <div class="dim-icon">⚖️</div>
          <div class="dim-name">Bias</div>
          <div class="dim-desc">Is the AI analysis unfairly skewed toward a group?</div>
        </div>
        <div class="dim">
          <div class="dim-icon">📋</div>
          <div class="dim-name">Alignment</div>
          <div class="dim-desc">Did the AI correctly flag platform rule violations?</div>
        </div>
        <div class="dim">
          <div class="dim-icon">🧠</div>
          <div class="dim-name">Memory</div>
          <div class="dim-desc">Did AI consider the author's posting history?</div>
        </div>
      </div>
    </div>

    <!-- API Endpoints -->
    <div class="card">
      <div class="card-title">🔌 API Endpoints</div>
      <div class="endpoints">
        <div class="endpoint">
          <div class="ep-method">POST</div>
          <div class="ep-path">/reset</div>
          <div class="ep-desc">Start new episode, get first observation</div>
        </div>
        <div class="endpoint">
          <div class="ep-method">POST</div>
          <div class="ep-path">/step</div>
          <div class="ep-desc">Submit audit action, get reward + next obs</div>
        </div>
        <div class="endpoint">
          <div class="ep-method">GET</div>
          <div class="ep-path">/state</div>
          <div class="ep-desc">Get current episode state and history</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tasks -->
  <div class="card" style="margin-bottom:24px">
    <div class="card-title">📋 Task Scenarios</div>
""" + "".join([f"""
    <div class="task-card">
      <div class="task-header">
        <span class="task-badge badge-{t['id']}">{t['id'].upper()}</span>
        <span class="verdict-badge verdict-{t['verdict']}">Expected: {t['verdict']}</span>
      </div>
      <div class="task-author">@{t['author']}</div>
      <div class="task-content">{t['content']}</div>
    </div>
""" for t in tasks_preview]) + """
  </div>

</div>

<footer>
  Built for Meta PyTorch OpenEnv Hackathon 2026 · Social Media AI Auditor v1.0
</footer>

<script>
let totalReward = 0;
let stepsDone = 0;

async function doReset() {
  setLoading('Resetting environment...');
  totalReward = 0; stepsDone = 0;
  updateScore(0, 0, {});
  try {
    const r = await fetch('/reset', {method:'POST'});
    const d = await r.json();
    document.getElementById('response').className = 'response-box success';
    document.getElementById('response').textContent = JSON.stringify(d, null, 2);
    document.getElementById('step-btn').disabled = false;
    document.getElementById('score-section').style.display = 'block';
  } catch(e) { setError(e.message); }
}

async function doStep() {
  setLoading('Agent analyzing post...');
  try {
    // Auto-agent — sends a realistic audit action
    const action = {
      hallucination_detected: true,
      hallucination_explanation: "The AI analysis contains factual errors not supported by evidence",
      bias_detected: false,
      bias_explanation: "No significant bias detected in this post",
      alignment_violated: true,
      alignment_explanation: "Post violates platform rules on misinformation",
      memory_consistent: true,
      memory_explanation: "Author history is consistent with current post",
      overall_verdict: "remove",
      confidence: 0.8
    };
    const r = await fetch('/step', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(action)
    });
    const d = await r.json();
    totalReward += d.reward || 0;
    stepsDone += 1;
    updateScore(totalReward, stepsDone, d.info?.breakdown || {});
    document.getElementById('response').className = 'response-box success';
    document.getElementById('response').textContent = JSON.stringify(d, null, 2);
    if (d.done) {
      document.getElementById('step-btn').disabled = true;
      document.getElementById('step-btn').textContent = 'Episode Complete — Click Reset';
    }
  } catch(e) { setError(e.message); }
}

async function doState() {
  setLoading('Fetching state...');
  try {
    const r = await fetch('/state');
    const d = await r.json();
    document.getElementById('response').className = 'response-box';
    document.getElementById('response').textContent = JSON.stringify(d, null, 2);
  } catch(e) { setError(e.message); }
}

async function doHealth() {
  setLoading('Checking health...');
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('response').className = 'response-box success';
    document.getElementById('response').textContent = JSON.stringify(d, null, 2);
  } catch(e) { setError(e.message); }
}

function setLoading(msg) {
  document.getElementById('response').className = 'response-box loading';
  document.getElementById('response').textContent = msg;
}

function setError(msg) {
  document.getElementById('response').className = 'response-box error';
  document.getElementById('response').textContent = 'Error: ' + msg;
}

function updateScore(reward, steps, breakdown) {
  const pct = Math.min((reward / 3.0) * 100, 100);
  document.getElementById('score-num').textContent = reward.toFixed(3);
  document.getElementById('score-fill').style.width = pct + '%';
  document.getElementById('score-steps').textContent = steps + ' / 3 tasks';
  document.getElementById('score-section').style.display = 'block';

  const labels = {hallucination:'Hallucination', bias:'Bias', alignment:'Alignment', memory:'Memory', verdict:'Verdict'};
  const bd = document.getElementById('breakdown');
  bd.innerHTML = Object.entries(labels).map(([k,l]) => `
    <div class="rb-item">
      <div class="rb-val">${breakdown[k] !== undefined ? breakdown[k].toFixed(2) : '—'}</div>
      <div class="rb-label">${l}</div>
    </div>
  `).join('');
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)