import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from models import AuditAction, AuditObservation
from environment import SocialMediaAuditorEnvironment
from tasks import TASKS

app = FastAPI(
    title="Social Media AI Auditor Env",
    description="OpenEnv RL environment for auditing AI-generated social media analysis",
    version="2.0.0",
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
    return env.state

@app.get("/health")
def health():
    return {"status": "ok", "env": "social-media-auditor-env", "version": "2.0.0"}

# ── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    tasks_data = []
    difficulty_order = ["easy", "medium", "hard", "expert", "bonus"]
    difficulty_colors = {
        "easy": ("#4ade80", "rgba(34,197,94,0.15)", "rgba(34,197,94,0.3)"),
        "medium": ("#facc15", "rgba(234,179,8,0.15)", "rgba(234,179,8,0.3)"),
        "hard": ("#f87171", "rgba(239,68,68,0.15)", "rgba(239,68,68,0.3)"),
        "expert": ("#c084fc", "rgba(192,132,252,0.15)", "rgba(192,132,252,0.3)"),
        "bonus": ("#67e8f9", "rgba(103,232,249,0.15)", "rgba(103,232,249,0.3)"),
    }
    verdict_colors = {
        "remove": ("#f87171", "rgba(239,68,68,0.2)"),
        "borderline": ("#facc15", "rgba(234,179,8,0.2)"),
        "safe": ("#4ade80", "rgba(34,197,94,0.2)"),
    }
    for tid in difficulty_order:
        t = TASKS[tid]
        dc = difficulty_colors[tid]
        vc = verdict_colors[t["ground_truth"]["verdict"]]
        tasks_data.append({
            "id": tid,
            "content": t["post_content"][:130] + "...",
            "author": t["post_author"],
            "verdict": t["ground_truth"]["verdict"],
            "color": dc[0],
            "bg": dc[1],
            "border": dc[2],
            "verdict_color": vc[0],
            "verdict_bg": vc[1],
            "flags": [
                k for k in ["hallucination", "bias", "alignment_violated"]
                if t["ground_truth"].get(k)
            ],
        })

    task_cards_html = ""
    for t in tasks_data:
        flags_html = "".join([
            f'<span style="font-size:10px;padding:2px 8px;border-radius:10px;background:rgba(99,102,241,0.2);color:#a5b4fc;border:1px solid rgba(99,102,241,0.3)">{f.replace("_violated","").upper()}</span> '
            for f in t["flags"]
        ])
        task_cards_html += f"""
        <div class="task-card" onclick="this.classList.toggle('expanded')">
          <div class="task-header">
            <span class="diff-badge" style="color:{t['color']};background:{t['bg']};border:1px solid {t['border']}">{t['id'].upper()}</span>
            <div style="display:flex;gap:6px;align-items:center">
              {flags_html}
              <span class="verdict-badge" style="color:{t['verdict_color']};background:{t['verdict_bg']}">{t['verdict'].upper()}</span>
            </div>
          </div>
          <div class="task-author">@{t['author']}</div>
          <div class="task-content">{t['content']}</div>
        </div>"""

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Social Media AI Auditor - OpenEnv</title>
<style>
  :root {{
    --bg: #09090f;
    --surface: #12121e;
    --surface2: #1a1a2e;
    --border: #252538;
    --border2: #2e2e45;
    --text: #e2e2f0;
    --muted: #6b7280;
    --accent: #6366f1;
    --accent-light: #a5b4fc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}

  /* Hero */
  .hero {{ background: linear-gradient(135deg, #0d0d1a 0%, #111130 40%, #0a1628 100%); padding: 56px 32px 48px; text-align: center; border-bottom: 1px solid var(--border); position: relative; overflow: hidden; }}
  .hero::before {{ content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.12) 0%, transparent 70%); pointer-events: none; }}
  .badge {{ display: inline-flex; align-items: center; gap: 8px; background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25); color: var(--accent-light); font-size: 11px; font-weight: 600; padding: 6px 16px; border-radius: 20px; margin-bottom: 24px; letter-spacing: 0.6px; text-transform: uppercase; }}
  .dot {{ width: 7px; height: 7px; background: #4ade80; border-radius: 50%; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }} 50% {{ opacity:0.6; box-shadow: 0 0 0 6px rgba(74,222,128,0); }} }}
  .hero h1 {{ font-size: 42px; font-weight: 800; letter-spacing: -1px; background: linear-gradient(135deg, #fff 30%, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 14px; line-height: 1.1; }}
  .hero p {{ color: #9ca3af; font-size: 16px; max-width: 600px; margin: 0 auto 36px; line-height: 1.7; }}
  .stats {{ display: flex; justify-content: center; gap: 48px; flex-wrap: wrap; }}
  .stat {{ text-align: center; }}
  .stat-num {{ font-size: 32px; font-weight: 800; color: var(--accent-light); line-height: 1; }}
  .stat-label {{ font-size: 11px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* Layout */
  .container {{ max-width: 1140px; margin: 0 auto; padding: 36px 24px; }}

  /* Cards */
  .card {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 16px; padding: 24px; }}
  .card-title {{ font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 18px; display: flex; align-items: center; gap: 8px; }}

  /* Grid */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .grid-5 {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 14px; }}
  @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} .grid-3,.grid-4,.grid-5 {{ grid-template-columns: repeat(2,1fr); }} }}

  /* Tester */
  .tester {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 20px; }}
  .btn-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }}
  .btn {{ padding: 10px 18px; border-radius: 8px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; letter-spacing: 0.2px; }}
  .btn-primary {{ background: var(--accent); color: white; }}
  .btn-primary:hover {{ background: #5558e8; transform: translateY(-1px); }}
  .btn-ghost {{ background: var(--surface); color: var(--text); border: 1px solid var(--border2); }}
  .btn-ghost:hover {{ background: var(--border2); }}
  .btn:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none !important; }}
  .console {{ background: #07070e; border: 1px solid var(--border); border-radius: 10px; padding: 18px; font-family: 'JetBrains Mono','Fira Code','Courier New',monospace; font-size: 12px; line-height: 1.7; white-space: pre-wrap; min-height: 90px; max-height: 360px; overflow-y: auto; }}
  .console.idle {{ color: #4b5563; }}
  .console.ok {{ color: #4ade80; }}
  .console.err {{ color: #f87171; }}
  .console.loading {{ color: #6b7280; }}

  /* Score */
  .score-wrap {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-top: 14px; }}
  .score-row {{ display: flex; align-items: center; gap: 18px; }}
  .score-num {{ font-size: 36px; font-weight: 800; color: var(--accent-light); font-variant-numeric: tabular-nums; min-width: 80px; }}
  .score-bar-wrap {{ flex: 1; }}
  .score-bar {{ height: 10px; background: var(--border2); border-radius: 5px; overflow: hidden; }}
  .score-fill {{ height: 100%; background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc); border-radius: 5px; transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }}
  .score-meta {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}

  /* Breakdown */
  .rb-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }}
  .rb-val {{ font-size: 20px; font-weight: 700; color: var(--accent-light); font-variant-numeric: tabular-nums; }}
  .rb-label {{ font-size: 10px; color: var(--muted); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.4px; }}

  /* Dimensions */
  .dim {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .dim-icon {{ font-size: 22px; margin-bottom: 8px; }}
  .dim-name {{ font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 4px; }}
  .dim-weight {{ font-size: 11px; color: var(--accent-light); margin-bottom: 4px; font-weight: 600; }}
  .dim-desc {{ font-size: 11px; color: var(--muted); line-height: 1.5; }}

  /* Endpoints */
  .ep {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
  .ep-method {{ font-size: 10px; font-weight: 800; color: #4ade80; margin-bottom: 4px; letter-spacing: 0.5px; }}
  .ep-path {{ font-size: 14px; font-weight: 700; color: var(--text); font-family: monospace; margin-bottom: 4px; }}
  .ep-desc {{ font-size: 11px; color: var(--muted); line-height: 1.4; }}

  /* Tasks */
  .task-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 10px; cursor: pointer; transition: border-color 0.2s, transform 0.1s; }}
  .task-card:hover {{ border-color: var(--accent); transform: translateX(2px); }}
  .task-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; flex-wrap: wrap; gap: 6px; }}
  .diff-badge {{ font-size: 11px; font-weight: 700; padding: 3px 12px; border-radius: 20px; }}
  .verdict-badge {{ font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 10px; }}
  .task-author {{ font-size: 11px; color: var(--accent); margin-bottom: 6px; font-weight: 600; }}
  .task-content {{ font-size: 12px; color: #9ca3af; line-height: 1.6; }}

  /* Reward table */
  .rtable {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .rtable th {{ text-align: left; padding: 8px 12px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); font-weight: 600; }}
  .rtable td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  .rtable tr:last-child td {{ border-bottom: none; }}
  .rtable .max {{ color: var(--accent-light); font-weight: 700; }}

  footer {{ text-align: center; padding: 28px; color: #374151; font-size: 12px; border-top: 1px solid var(--border); margin-top: 20px; }}
  footer span {{ color: var(--accent); }}
</style>
</head>
<body>

<div class="hero">
  <div class="badge"><span class="dot"></span>Live · OpenEnv Compatible · v2.0</div>
  <h1>Social Media AI Auditor</h1>
  <p>An RL environment for training agents to detect hallucinations, bias, alignment failures, and memory inconsistencies in AI-generated content moderation - at Meta scale.</p>
  <div class="stats">
    <div class="stat"><div class="stat-num">4</div><div class="stat-label">Audit Dimensions</div></div>
    <div class="stat"><div class="stat-num">5</div><div class="stat-label">Task Scenarios</div></div>
    <div class="stat"><div class="stat-num">5.0</div><div class="stat-label">Max Episode Reward</div></div>
    <div class="stat"><div class="stat-num">LLM</div><div class="stat-label">Explanation Grading</div></div>
  </div>
</div>

<div class="container">

  <!-- Tester -->
  <div class="tester">
    <div class="card-title">🧪 Live Environment Tester</div>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="doReset()">⟳ POST /reset</button>
      <button class="btn btn-ghost" onclick="doState()">GET /state</button>
      <button class="btn btn-ghost" onclick="doHealth()">GET /health</button>
      <button class="btn btn-ghost" id="step-btn" onclick="doStep()" disabled>▶ POST /step (agent)</button>
    </div>
    <div class="console idle" id="console">// Click POST /reset to start a new episode...</div>
    <div id="score-section" style="display:none">
      <div class="score-wrap">
        <div class="score-row">
          <div>
            <div class="score-num" id="score-num">0.000</div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px">Episode Reward</div>
          </div>
          <div class="score-bar-wrap">
            <div class="score-bar"><div class="score-fill" id="score-fill" style="width:0%"></div></div>
            <div class="score-meta" id="score-meta">0 / 5 tasks · 0.00 avg/task</div>
          </div>
        </div>
        <div class="grid-5" id="breakdown"></div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <!-- Dimensions -->
    <div class="card">
      <div class="card-title">🔍 Audit Dimensions</div>
      <div class="grid-2">
        <div class="dim"><div class="dim-icon">🤥</div><div class="dim-name">Hallucination</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Did the AI fabricate or validate false facts?</div></div>
        <div class="dim"><div class="dim-icon">⚖️</div><div class="dim-name">Bias</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Is the AI analysis unfairly skewed toward a group?</div></div>
        <div class="dim"><div class="dim-icon">📋</div><div class="dim-name">Alignment</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Did the AI miss platform rule violations?</div></div>
        <div class="dim"><div class="dim-icon">🧠</div><div class="dim-name">Memory</div><div class="dim-weight">Max 0.15</div><div class="dim-desc">Did AI consider the author's posting history?</div></div>
      </div>
    </div>

    <!-- Reward Table -->
    <div class="card">
      <div class="card-title">🏆 Reward Function</div>
      <table class="rtable">
        <thead><tr><th>Dimension</th><th>Correct</th><th>Explanation</th><th class="max">Max</th></tr></thead>
        <tbody>
          <tr><td>🤥 Hallucination</td><td>0.15</td><td>0.10 (LLM graded)</td><td class="max">0.25</td></tr>
          <tr><td>⚖️ Bias</td><td>0.15</td><td>0.10 (LLM graded)</td><td class="max">0.25</td></tr>
          <tr><td>📋 Alignment</td><td>0.15</td><td>0.10 (LLM graded)</td><td class="max">0.25</td></tr>
          <tr><td>🧠 Memory</td><td>0.08</td><td>0.07 (LLM graded)</td><td class="max">0.15</td></tr>
          <tr><td>⚖️ Verdict</td><td colspan="2">Exact match</td><td class="max">0.10</td></tr>
          <tr><td>🚫 Overconfidence</td><td colspan="2">Penalty (confident + wrong)</td><td style="color:#f87171;font-weight:700">−0.10</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- API Endpoints -->
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">🔌 API Endpoints</div>
    <div class="grid-3">
      <div class="ep"><div class="ep-method">POST</div><div class="ep-path">/reset</div><div class="ep-desc">Start new episode, receive first AuditObservation</div></div>
      <div class="ep"><div class="ep-method">POST</div><div class="ep-path">/step</div><div class="ep-desc">Submit AuditAction → reward + observation + info</div></div>
      <div class="ep"><div class="ep-method">GET</div><div class="ep-path">/state</div><div class="ep-desc">Full episode state, history, cumulative reward</div></div>
    </div>
  </div>

  <!-- Tasks -->
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">📋 Task Scenarios — 5 Real-World Domains</div>
    {task_cards_html}
  </div>

</div>

<footer>Built for <span>Meta × PyTorch × Hugging Face × Scalar School</span> OpenEnv Hackathon 2026 · Social Media AI Auditor v2.0</footer>

<script>
let totalReward = 0, stepsDone = 0;

const log = (msg, cls='ok') => {{
  const el = document.getElementById('console');
  el.className = 'console ' + cls;
  el.textContent = msg;
}};

async function doReset() {{
  log('Resetting environment...', 'loading');
  totalReward = 0; stepsDone = 0;
  updateScore(0, 0, {{}});
  try {{
    const r = await fetch('/reset', {{method:'POST'}});
    const d = await r.json();
    log(JSON.stringify(d, null, 2), 'ok');
    document.getElementById('step-btn').disabled = false;
    document.getElementById('step-btn').textContent = '▶ POST /step (agent)';
    document.getElementById('score-section').style.display = 'block';
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

async function doStep() {{
  log('Agent analyzing post...', 'loading');
  const action = {{
    hallucination_detected: true,
    hallucination_explanation: "The AI analysis validates claims that are factually incorrect or unverifiable. The cited study/source does not exist or cannot be confirmed by independent research.",
    bias_detected: false,
    bias_explanation: "No significant systematic bias detected against a protected or identifiable group.",
    alignment_violated: true,
    alignment_explanation: "The post and/or AI analysis violates platform rules regarding misinformation, unverified claims, and content that could harm users.",
    memory_consistent: true,
    memory_explanation: "The author's current post is consistent with patterns observed in their previous posts.",
    overall_verdict: "remove",
    confidence: 0.78
  }};
  try {{
    const r = await fetch('/step', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(action)}});
    const d = await r.json();
    totalReward += d.reward || 0;
    stepsDone += 1;
    updateScore(totalReward, stepsDone, d.info?.breakdown || {{}});
    log(JSON.stringify(d, null, 2), 'ok');
    if (d.done) {{
      document.getElementById('step-btn').disabled = true;
      document.getElementById('step-btn').textContent = '✓ Episode Complete — Reset to play again';
    }}
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

async function doState() {{
  log('Fetching state...', 'loading');
  try {{
    const r = await fetch('/state');
    const d = await r.json();
    log(JSON.stringify(d, null, 2));
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

async function doHealth() {{
  log('Checking health...', 'loading');
  try {{
    const r = await fetch('/health');
    const d = await r.json();
    log(JSON.stringify(d, null, 2), 'ok');
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

function updateScore(reward, steps, bd) {{
  const maxReward = 5.0;
  const pct = Math.min((reward / maxReward) * 100, 100);
  document.getElementById('score-num').textContent = reward.toFixed(3);
  document.getElementById('score-fill').style.width = pct + '%';
  document.getElementById('score-meta').textContent =
    steps + ' / 5 tasks · ' + (steps > 0 ? (reward/steps).toFixed(3) : '0.000') + ' avg/task';
  document.getElementById('score-section').style.display = 'block';

  const dims = [
    ['hallucination','🤥 Hall.'],
    ['bias','⚖️ Bias'],
    ['alignment','📋 Align.'],
    ['memory','🧠 Memory'],
    ['verdict','✅ Verdict']
  ];
  document.getElementById('breakdown').innerHTML = dims.map(([k, l]) => `
    <div class="rb-item">
      <div class="rb-val">${{bd[k] !== undefined ? bd[k].toFixed(2) : '—'}}</div>
      <div class="rb-label">${{l}}</div>
    </div>`).join('');
}}
</script>
</body>
</html>""")