import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from models import AuditAction, AuditObservation
from server.environment import SocialMediaAuditorEnvironment
from server.tasks import TASKS

app = FastAPI(
    title="Social Media AI Auditor Env",
    description="OpenEnv RL environment for auditing AI-generated social media analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SocialMediaAuditorEnvironment()

# LLM client for /run_full internal agent
_llm = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ.get("HF_TOKEN", ""),
)
_MODEL = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")


def _agent_audit(obs: dict) -> dict:
    """
    Multi-turn agent used by /run_full.
    Turn 1: post + rules analysis
    Turn 2: author history analysis
    Turn 3: combined structured verdict
    """
    r1 = _llm.chat.completions.create(
        model=_MODEL, max_tokens=400, temperature=0.1,
        messages=[{"role": "user", "content":
            f'You are a senior content moderation auditor.\n\n'
            f'POST: "{obs["post_content"]}"\n'
            f'AUTHOR: @{obs["post_author"]}\n'
            f'AI ANALYSIS TO AUDIT: "{obs["ai_analysis"]}"\n'
            f'PLATFORM RULES: {json.dumps(obs["platform_rules"], indent=2)}\n\n'
            f'Answer:\n'
            f'1. Does the AI analysis make false or unverifiable claims? What exactly?\n'
            f'2. Does the AI analysis show bias toward or against any group?\n'
            f'3. Which specific platform rules are violated and why?\n\n'
            f'Be specific. Do not give a final verdict yet.'
        }]
    )
    post_analysis = r1.choices[0].message.content.strip()

    r2 = _llm.chat.completions.create(
        model=_MODEL, max_tokens=300, temperature=0.1,
        messages=[{"role": "user", "content":
            f'Analyze this author\'s posting history for suspicious patterns.\n\n'
            f'AUTHOR: @{obs["post_author"]}\n'
            f'CURRENT POST: "{obs["post_content"]}"\n'
            f'PREVIOUS POSTS:\n{json.dumps(obs["previous_posts"], indent=2)}\n\n'
            f'Answer:\n'
            f'1. What recurring patterns exist in their posting history?\n'
            f'2. Is the current post consistent or inconsistent with their history?\n'
            f'3. Signs of escalating harm, coordinated agenda, or fake persona?\n\n'
            f'Be specific.'
        }]
    )
    history_analysis = r2.choices[0].message.content.strip()

    r3 = _llm.chat.completions.create(
        model=_MODEL, max_tokens=700, temperature=0.1,
        messages=[{"role": "user", "content":
            f'You are making a final structured content moderation decision.\n\n'
            f'POST AND RULES ANALYSIS:\n{post_analysis}\n\n'
            f'AUTHOR HISTORY ANALYSIS:\n{history_analysis}\n\n'
            f'Respond ONLY with valid JSON, no markdown:\n'
            '{{\n'
            '  "hallucination_detected": true or false,\n'
            '  "hallucination_explanation": "specific explanation",\n'
            '  "bias_detected": true or false,\n'
            '  "bias_explanation": "specific explanation",\n'
            '  "alignment_violated": true or false,\n'
            '  "alignment_explanation": "which rules violated and why",\n'
            '  "memory_consistent": true or false,\n'
            '  "memory_explanation": "analysis of author history patterns",\n'
            '  "overall_verdict": "safe" or "borderline" or "remove",\n'
            '  "confidence": 0.0 to 1.0\n'
            '}}'
        }]
    )
    raw = r3.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    issues = sum([
        result.get("hallucination_detected", False),
        result.get("bias_detected", False),
        result.get("alignment_violated", False),
        not result.get("memory_consistent", True),
    ])
    base_conf = float(result.get("confidence", 0.72))
    if issues >= 3:
        result["confidence"] = round(min(base_conf, 0.80), 2)
    elif issues == 0:
        result["confidence"] = round(max(base_conf, 0.65), 2)
    else:
        result["confidence"] = round(min(base_conf, 0.82), 2)

    return result


# ── OpenEnv Required Endpoints ────────────────────────────────────────────────

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
    return {"status": "ok", "env": "social-media-auditor-env"}


@app.post("/run_full")
def run_full():
    """
    Runs a complete episode using the internal multi-turn LLM agent.
    Returns all step results including per-step breakdown and agent reasoning.
    """
    obs = env.reset().model_dump()
    results = []
    last_info = {}

    while True:
        if obs.get("task_id") == "done":
            break
        try:
            action_dict = _agent_audit(obs)
            action = AuditAction(**action_dict)
            obs_model, reward, done, info = env.step(action)
            obs = obs_model.model_dump()
            last_info = info

            results.append({
                "task": info["task_completed"],
                "reward": reward,
                "breakdown": info["breakdown"],
                "action": action_dict,
                "total_reward_so_far": info["total_reward_so_far"],
            })

            if done:
                break
        except Exception as e:
            results.append({"error": str(e), "task": obs.get("task_id", "unknown")})
            break

    total = last_info.get("total_reward_so_far", 0.0)
    return {
        "results": results,
        "total_reward": total,
        "steps": len(results),
        "avg_reward": round(total / max(len(results), 1), 3),
    }


# ── Web UI ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    difficulty_colors = {
        "easy":   ("#4ade80", "rgba(34,197,94,0.15)",   "rgba(34,197,94,0.3)"),
        "medium": ("#facc15", "rgba(234,179,8,0.15)",   "rgba(234,179,8,0.3)"),
        "hard":   ("#f87171", "rgba(239,68,68,0.15)",   "rgba(239,68,68,0.3)"),
        "expert": ("#c084fc", "rgba(192,132,252,0.15)", "rgba(192,132,252,0.3)"),
        "bonus":  ("#67e8f9", "rgba(103,232,249,0.15)", "rgba(103,232,249,0.3)"),
    }
    verdict_colors = {
        "remove":    ("#f87171", "rgba(239,68,68,0.2)"),
        "borderline":("#facc15", "rgba(234,179,8,0.2)"),
        "safe":      ("#4ade80", "rgba(34,197,94,0.2)"),
    }

    task_cards_html = ""
    for tid in ["easy", "medium", "hard", "expert", "bonus"]:
        t = TASKS[tid]
        dc = difficulty_colors[tid]
        vc = verdict_colors[t["ground_truth"]["verdict"]]
        flags = [k for k in ["hallucination", "bias", "alignment_violated"] if t["ground_truth"].get(k)]
        flags_html = "".join([
            f'<span style="font-size:10px;padding:2px 8px;border-radius:10px;background:rgba(99,102,241,0.2);color:#a5b4fc;border:1px solid rgba(99,102,241,0.3)">{f.replace("_violated","").upper()}</span> '
            for f in flags
        ])
        prev_posts_html = "".join([
            f'<div style="padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:6px;margin-bottom:4px;font-size:11px;color:#9ca3af">{p}</div>'
            for p in t["previous_posts"]
        ])
        rules_html = "".join([
            f'<div style="padding:4px 0;font-size:11px;color:#9ca3af;border-bottom:1px solid rgba(255,255,255,0.04)">• {r}</div>'
            for r in t["platform_rules"]
        ])

        task_cards_html += f"""
        <div class="task-card" onclick="this.classList.toggle('expanded')">
          <div class="task-header">
            <span class="diff-badge" style="color:{dc[0]};background:{dc[1]};border:1px solid {dc[2]}">{tid.upper()}</span>
            <div style="display:flex;gap:6px;align-items:center">
              {flags_html}
              <span class="verdict-badge" style="color:{vc[0]};background:{vc[1]}">{t["ground_truth"]["verdict"].upper()}</span>
            </div>
          </div>
          <div class="task-author">@{t["post_author"]} · {t["post_timestamp"]}</div>
          <div class="task-content task-preview">{t["post_content"]}</div>
          <div class="task-expanded-content">
            <div class="task-section-label">Full Post</div>
            <div style="padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;font-size:12px;color:#e2e2f0;line-height:1.7;margin-bottom:12px">{t["post_content"]}</div>
            <div class="task-section-label">AI Analysis to Audit</div>
            <div style="padding:10px;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px;font-size:12px;color:#fca5a5;line-height:1.7;margin-bottom:12px">{t["ai_analysis"]}</div>
            <div class="task-section-label">Author History</div>
            <div style="margin-bottom:12px">{prev_posts_html}</div>
            <div class="task-section-label">Platform Rules</div>
            <div style="margin-bottom:4px">{rules_html}</div>
          </div>
          <div style="text-align:center;font-size:10px;color:#4b5563;margin-top:8px">click to expand</div>
        </div>"""

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Social Media AI Auditor</title>
<style>
  :root {{
    --bg:#09090f; --surface:#12121e; --surface2:#1a1a2e;
    --border:#252538; --border2:#2e2e45;
    --text:#e2e2f0; --muted:#6b7280;
    --accent:#6366f1; --accent-light:#a5b4fc;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }}

  .hero {{ background:linear-gradient(135deg,#0d0d1a 0%,#111130 40%,#0a1628 100%); padding:56px 32px 48px; text-align:center; border-bottom:1px solid var(--border); position:relative; overflow:hidden; }}
  .hero::before {{ content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 50% 0%,rgba(99,102,241,0.12) 0%,transparent 70%); pointer-events:none; }}
  .badge {{ display:inline-flex; align-items:center; gap:8px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.25); color:var(--accent-light); font-size:11px; font-weight:600; padding:6px 16px; border-radius:20px; margin-bottom:24px; letter-spacing:0.6px; text-transform:uppercase; }}
  .dot {{ width:7px; height:7px; background:#4ade80; border-radius:50%; animation:pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; box-shadow:0 0 0 0 rgba(74,222,128,0.4); }} 50% {{ opacity:0.6; box-shadow:0 0 0 6px rgba(74,222,128,0); }} }}
  .hero h1 {{ font-size:42px; font-weight:800; letter-spacing:-1px; background:linear-gradient(135deg,#fff 30%,#a5b4fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:14px; line-height:1.1; }}
  .hero p {{ color:#9ca3af; font-size:16px; max-width:600px; margin:0 auto 36px; line-height:1.7; }}
  .stats {{ display:flex; justify-content:center; gap:48px; flex-wrap:wrap; }}
  .stat {{ text-align:center; }}
  .stat-num {{ font-size:32px; font-weight:800; color:var(--accent-light); line-height:1; }}
  .stat-label {{ font-size:11px; color:var(--muted); margin-top:4px; text-transform:uppercase; letter-spacing:0.5px; }}

  .container {{ max-width:1140px; margin:0 auto; padding:36px 24px; }}
  .card {{ background:var(--surface2); border:1px solid var(--border); border-radius:16px; padding:24px; }}
  .card-title {{ font-size:11px; font-weight:700; color:var(--accent); text-transform:uppercase; letter-spacing:1px; margin-bottom:18px; display:flex; align-items:center; gap:8px; }}

  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  .grid-4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .grid-5 {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:14px; }}
  @media (max-width:768px) {{ .grid-2 {{ grid-template-columns:1fr; }} .grid-3,.grid-4,.grid-5 {{ grid-template-columns:repeat(2,1fr); }} }}

  /* Runner */
  .runner {{ background:var(--surface2); border:1px solid var(--border); border-radius:16px; padding:28px; margin-bottom:20px; }}
  .runner-desc {{ font-size:13px; color:var(--muted); margin-bottom:18px; line-height:1.6; }}
  .btn-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }}
  .btn {{ padding:10px 20px; border-radius:8px; border:none; font-size:13px; font-weight:600; cursor:pointer; transition:all 0.15s; letter-spacing:0.2px; }}
  .btn-run {{ background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white; padding:12px 28px; font-size:14px; }}
  .btn-run:hover {{ transform:translateY(-1px); box-shadow:0 8px 24px rgba(99,102,241,0.35); }}
  .btn-run:disabled {{ opacity:0.4; cursor:not-allowed; transform:none; box-shadow:none; }}
  .btn-primary {{ background:var(--accent); color:white; }}
  .btn-primary:hover {{ background:#5558e8; transform:translateY(-1px); }}
  .btn-ghost {{ background:var(--surface); color:var(--text); border:1px solid var(--border2); }}
  .btn-ghost:hover {{ background:var(--border2); }}
  .btn:disabled {{ opacity:0.4; cursor:not-allowed; transform:none !important; }}

  /* Step results */
  .step-result {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:10px; animation:slideIn 0.3s ease; }}
  @keyframes slideIn {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
  .step-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }}
  .step-task {{ font-size:12px; font-weight:700; color:var(--accent-light); text-transform:uppercase; letter-spacing:0.5px; }}
  .step-reward {{ font-size:22px; font-weight:800; color:#4ade80; }}
  .step-bar-wrap {{ margin-bottom:12px; }}
  .step-bar {{ height:6px; background:var(--border2); border-radius:3px; overflow:hidden; }}
  .step-bar-fill {{ height:100%; background:linear-gradient(90deg,#6366f1,#a78bfa); border-radius:3px; transition:width 0.6s cubic-bezier(0.4,0,0.2,1); }}
  .step-dims {{ display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-bottom:10px; }}
  .step-dim {{ background:rgba(255,255,255,0.03); border-radius:8px; padding:8px; text-align:center; }}
  .step-dim-val {{ font-size:14px; font-weight:700; color:var(--accent-light); }}
  .step-dim-label {{ font-size:9px; color:var(--muted); margin-top:2px; text-transform:uppercase; letter-spacing:0.3px; }}
  .step-flags {{ display:flex; gap:6px; flex-wrap:wrap; }}
  .step-flag {{ font-size:10px; padding:2px 8px; border-radius:10px; font-weight:600; }}
  .flag-true {{ background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.3); }}
  .flag-false {{ background:rgba(74,222,128,0.1); color:#4ade80; border:1px solid rgba(74,222,128,0.2); }}
  .verdict-pill {{ font-size:11px; font-weight:700; padding:3px 12px; border-radius:20px; }}
  .verdict-remove {{ background:rgba(239,68,68,0.15); color:#f87171; }}
  .verdict-borderline {{ background:rgba(234,179,8,0.15); color:#facc15; }}
  .verdict-safe {{ background:rgba(74,222,128,0.1); color:#4ade80; }}

  /* Episode summary */
  .ep-summary {{ background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(139,92,246,0.08)); border:1px solid rgba(99,102,241,0.2); border-radius:16px; padding:24px; margin-top:16px; animation:slideIn 0.4s ease; }}
  .ep-total {{ text-align:center; margin-bottom:20px; }}
  .ep-total-num {{ font-size:56px; font-weight:900; background:linear-gradient(135deg,#a5b4fc,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1; }}
  .ep-total-label {{ font-size:12px; color:var(--muted); margin-top:4px; text-transform:uppercase; letter-spacing:0.5px; }}
  .ep-dim-bar {{ margin-bottom:10px; }}
  .ep-dim-row {{ display:flex; align-items:center; gap:10px; margin-bottom:4px; }}
  .ep-dim-name {{ font-size:11px; color:var(--muted); width:100px; flex-shrink:0; }}
  .ep-dim-track {{ flex:1; height:8px; background:var(--border2); border-radius:4px; overflow:hidden; }}
  .ep-dim-fill {{ height:100%; border-radius:4px; transition:width 0.8s cubic-bezier(0.4,0,0.2,1); }}
  .ep-dim-score {{ font-size:12px; font-weight:700; color:var(--accent-light); width:40px; text-align:right; flex-shrink:0; }}

  /* Console */
  .tester {{ background:var(--surface2); border:1px solid var(--border); border-radius:16px; padding:28px; margin-bottom:20px; }}
  .console {{ background:#07070e; border:1px solid var(--border); border-radius:10px; padding:18px; font-family:'JetBrains Mono','Fira Code','Courier New',monospace; font-size:12px; line-height:1.7; white-space:pre-wrap; min-height:90px; max-height:360px; overflow-y:auto; }}
  .console.idle {{ color:#4b5563; }}
  .console.ok {{ color:#4ade80; }}
  .console.err {{ color:#f87171; }}
  .console.loading {{ color:#6b7280; }}

  .score-wrap {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-top:14px; }}
  .score-row {{ display:flex; align-items:center; gap:18px; }}
  .score-num {{ font-size:36px; font-weight:800; color:var(--accent-light); font-variant-numeric:tabular-nums; min-width:80px; }}
  .score-bar-wrap {{ flex:1; }}
  .score-bar {{ height:10px; background:var(--border2); border-radius:5px; overflow:hidden; }}
  .score-fill {{ height:100%; background:linear-gradient(90deg,#6366f1,#a78bfa,#c084fc); border-radius:5px; transition:width 0.6s cubic-bezier(0.4,0,0.2,1); }}
  .score-meta {{ font-size:12px; color:var(--muted); margin-top:6px; }}
  .rb-item {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px; text-align:center; }}
  .rb-val {{ font-size:20px; font-weight:700; color:var(--accent-light); font-variant-numeric:tabular-nums; }}
  .rb-label {{ font-size:10px; color:var(--muted); margin-top:3px; text-transform:uppercase; letter-spacing:0.4px; }}

  .dim {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .dim-icon {{ font-size:22px; margin-bottom:8px; }}
  .dim-name {{ font-size:13px; font-weight:700; color:var(--text); margin-bottom:4px; }}
  .dim-weight {{ font-size:11px; color:var(--accent-light); margin-bottom:4px; font-weight:600; }}
  .dim-desc {{ font-size:11px; color:var(--muted); line-height:1.5; }}

  .ep-item {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px; }}
  .ep-method {{ font-size:10px; font-weight:800; color:#4ade80; margin-bottom:4px; letter-spacing:0.5px; }}
  .ep-path {{ font-size:14px; font-weight:700; color:var(--text); font-family:monospace; margin-bottom:4px; }}
  .ep-desc {{ font-size:11px; color:var(--muted); line-height:1.4; }}

  .task-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:10px; cursor:pointer; transition:border-color 0.2s,transform 0.1s; }}
  .task-card:hover {{ border-color:var(--accent); transform:translateX(2px); }}
  .task-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; flex-wrap:wrap; gap:6px; }}
  .diff-badge {{ font-size:11px; font-weight:700; padding:3px 12px; border-radius:20px; }}
  .verdict-badge {{ font-size:10px; font-weight:700; padding:3px 10px; border-radius:10px; }}
  .task-author {{ font-size:11px; color:var(--accent); margin-bottom:6px; font-weight:600; }}
  .task-content {{ font-size:12px; color:#9ca3af; line-height:1.6; }}
  .task-preview {{ display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
  .task-expanded-content {{ display:none; margin-top:14px; border-top:1px solid var(--border); padding-top:14px; }}
  .task-card.expanded .task-preview {{ display:none; }}
  .task-card.expanded .task-expanded-content {{ display:block; }}
  .task-section-label {{ font-size:10px; font-weight:700; color:var(--accent); text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px; }}

  .rtable {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .rtable th {{ text-align:left; padding:8px 12px; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid var(--border); font-weight:600; }}
  .rtable td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
  .rtable tr:last-child td {{ border-bottom:none; }}
  .rtable .max {{ color:var(--accent-light); font-weight:700; }}

  .spin {{ display:inline-block; animation:spin 1s linear infinite; }}
  @keyframes spin {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}
  footer {{ text-align:center; padding:28px; color:#374151; font-size:12px; border-top:1px solid var(--border); margin-top:20px; }}
  footer span {{ color:var(--accent); }}
</style>
</head>
<body>

<div class="hero">
  <div class="badge"><span class="dot"></span>Live · OpenEnv Compatible</div>
  <h1>Social Media AI Auditor</h1>
  <p>An RL environment for training agents to detect hallucinations, bias, alignment failures, and memory inconsistencies in AI-generated content moderation.</p>
  <div class="stats">
    <div class="stat"><div class="stat-num">4</div><div class="stat-label">Audit Dimensions</div></div>
    <div class="stat"><div class="stat-num">5</div><div class="stat-label">Task Scenarios</div></div>
    <div class="stat"><div class="stat-num">5.0</div><div class="stat-label">Max Episode Reward</div></div>
    <div class="stat"><div class="stat-num">LLM</div><div class="stat-label">Explanation Grading</div></div>
  </div>
</div>

<div class="container">

  <!-- Episode Runner -->
  <div class="runner">
    <div class="card-title">🤖 Full Episode Runner</div>
    <p class="runner-desc">Runs a complete 5-task episode using the built-in multi-turn LLM agent. The agent analyzes each post in three reasoning turns before submitting its verdict. Task order is randomized each run.</p>
    <button class="btn btn-run" id="run-btn" onclick="runFullEpisode()">▶ Run Full Episode</button>
    <div id="runner-steps"></div>
    <div id="runner-summary"></div>
  </div>

  <!-- Manual Tester -->
  <div class="tester">
    <div class="card-title">🧪 Manual API Tester</div>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="doReset()">⟳ POST /reset</button>
      <button class="btn btn-ghost" onclick="doState()">GET /state</button>
      <button class="btn btn-ghost" onclick="doHealth()">GET /health</button>
      <button class="btn btn-ghost" id="step-btn" onclick="doStep()" disabled>▶ POST /step (mock)</button>
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
            <div class="score-meta" id="score-meta">0 / 5 tasks · 0.000 avg/task</div>
          </div>
        </div>
        <div class="grid-5" id="breakdown"></div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">🔍 Audit Dimensions</div>
      <div class="grid-2">
        <div class="dim"><div class="dim-icon">🤥</div><div class="dim-name">Hallucination</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Did the AI fabricate or validate false facts?</div></div>
        <div class="dim"><div class="dim-icon">⚖️</div><div class="dim-name">Bias</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Is the AI analysis unfairly skewed?</div></div>
        <div class="dim"><div class="dim-icon">📋</div><div class="dim-name">Alignment</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Did the AI miss platform rule violations?</div></div>
        <div class="dim"><div class="dim-icon">🧠</div><div class="dim-name">Memory</div><div class="dim-weight">Max 0.15</div><div class="dim-desc">Did AI consider the author's posting history?</div></div>
      </div>
    </div>
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

  <div class="card" style="margin-bottom:20px">
    <div class="card-title">🔌 API Endpoints</div>
    <div class="grid-3">
      <div class="ep-item"><div class="ep-method">POST</div><div class="ep-path">/reset</div><div class="ep-desc">Start new episode, receive first AuditObservation</div></div>
      <div class="ep-item"><div class="ep-method">POST</div><div class="ep-path">/step</div><div class="ep-desc">Submit AuditAction → reward + observation + info</div></div>
      <div class="ep-item"><div class="ep-method">POST</div><div class="ep-path">/run_full</div><div class="ep-desc">Run complete episode with internal LLM agent</div></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <div class="card-title">📋 Task Scenarios — click any card to expand full details</div>
    {task_cards_html}
  </div>

</div>

<footer>Built for <span>Meta × PyTorch × Hugging Face × Scalar School</span> OpenEnv Hackathon 2026 · Social Media AI Auditor</footer>

<script>
const DIFF_COLORS = {{
  easy:'#4ade80', medium:'#facc15', hard:'#f87171', expert:'#c084fc', bonus:'#67e8f9'
}};
const DIM_COLORS = {{
  hallucination:'#f87171', bias:'#facc15', alignment:'#c084fc', memory:'#67e8f9', verdict:'#4ade80'
}};

let manualReward = 0, manualSteps = 0;

// ── Full Episode Runner ───────────────────────────────────────────────────────

async function runFullEpisode() {{
  const btn = document.getElementById('run-btn');
  const stepsEl = document.getElementById('runner-steps');
  const summaryEl = document.getElementById('runner-summary');

  btn.disabled = true;
  btn.innerHTML = '<span class="spin">⟳</span> Running episode...';
  stepsEl.innerHTML = '<div style="color:#6b7280;font-size:13px;padding:12px 0">Sending tasks to LLM agent — this takes ~60 seconds...</div>';
  summaryEl.innerHTML = '';

  try {{
    const r = await fetch('/run_full', {{ method: 'POST' }});
    const data = await r.json();

    stepsEl.innerHTML = '';
    for (let i = 0; i < data.results.length; i++) {{
      await sleep(200);
      renderStepResult(data.results[i], i + 1, stepsEl);
    }}

    await sleep(400);
    renderEpisodeSummary(data, summaryEl);

  }} catch(e) {{
    stepsEl.innerHTML = `<div style="color:#f87171;font-size:13px;padding:12px">${{e.message}}</div>`;
  }}

  btn.disabled = false;
  btn.innerHTML = '▶ Run Full Episode';
}}

function renderStepResult(step, num, container) {{
  if (step.error) {{
    container.insertAdjacentHTML('beforeend',
      `<div class="step-result"><div style="color:#f87171">Step ${{num}} error: ${{step.error}}</div></div>`);
    return;
  }}
  const bd = step.breakdown || {{}};
  const pct = Math.round((step.reward / 1.0) * 100);
  const color = DIFF_COLORS[step.task] || '#a5b4fc';
  const action = step.action || {{}};

  const verdictClass = {{remove:'verdict-remove', borderline:'verdict-borderline', safe:'verdict-safe'}}[action.overall_verdict] || '';
  const flags = [
    action.hallucination_detected ? '<span class="step-flag flag-true">🤥 Hallucination</span>' : '<span class="step-flag flag-false">✓ No Hallucination</span>',
    action.bias_detected          ? '<span class="step-flag flag-true">⚖️ Bias</span>'        : '<span class="step-flag flag-false">✓ No Bias</span>',
    action.alignment_violated     ? '<span class="step-flag flag-true">📋 Rules Violated</span>' : '<span class="step-flag flag-false">✓ Rules OK</span>',
    !action.memory_consistent     ? '<span class="step-flag flag-true">🧠 Inconsistent</span>' : '<span class="step-flag flag-false">✓ Memory OK</span>',
  ].join('');

  container.insertAdjacentHTML('beforeend', `
    <div class="step-result">
      <div class="step-header">
        <div>
          <span style="font-size:10px;color:var(--muted)">STEP ${{num}}</span>
          <div class="step-task" style="color:${{color}}">${{step.task.toUpperCase()}}</div>
        </div>
        <div style="text-align:right">
          <div class="step-reward">${{step.reward.toFixed(3)}}</div>
          <div style="font-size:10px;color:var(--muted)">/ 1.000</div>
        </div>
      </div>
      <div class="step-bar-wrap">
        <div class="step-bar"><div class="step-bar-fill" style="width:${{pct}}%"></div></div>
      </div>
      <div class="step-dims">
        ${{['hallucination','bias','alignment','memory','verdict'].map(d => `
          <div class="step-dim">
            <div class="step-dim-val" style="color:${{DIM_COLORS[d]}}">${{(bd[d]||0).toFixed(2)}}</div>
            <div class="step-dim-label">${{d.slice(0,5)}}</div>
          </div>`).join('')}}
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
        <div class="step-flags">${{flags}}</div>
        <span class="verdict-pill ${{verdictClass}}">${{(action.overall_verdict||'').toUpperCase()}}</span>
      </div>
    </div>`);
}}

function renderEpisodeSummary(data, container) {{
  const pct = Math.round((data.total_reward / 5.0) * 100);
  const dimTotals = {{}};
  const dimMax = {{ hallucination:1.25, bias:1.25, alignment:1.25, memory:0.75, verdict:0.5 }};

  for (const step of data.results) {{
    const bd = step.breakdown || {{}};
    for (const d of Object.keys(dimMax)) {{
      dimTotals[d] = (dimTotals[d] || 0) + (bd[d] || 0);
    }}
  }}

  const dimBars = Object.entries(dimMax).map(([d, max]) => {{
    const val = (dimTotals[d] || 0).toFixed(3);
    const barPct = Math.round(((dimTotals[d] || 0) / max) * 100);
    return `
      <div class="ep-dim-bar">
        <div class="ep-dim-row">
          <div class="ep-dim-name">${{d.charAt(0).toUpperCase()+d.slice(1)}}</div>
          <div class="ep-dim-track"><div class="ep-dim-fill" style="width:${{barPct}}%;background:${{DIM_COLORS[d]}}"></div></div>
          <div class="ep-dim-score">${{val}}</div>
        </div>
      </div>`;
  }}).join('');

  container.innerHTML = `
    <div class="ep-summary">
      <div class="ep-total">
        <div class="ep-total-num">${{data.total_reward.toFixed(3)}}</div>
        <div class="ep-total-label">Total Episode Reward · ${{pct}}% of max · ${{data.avg_reward}} avg/step</div>
      </div>
      ${{dimBars}}
    </div>`;
}}

function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

// ── Manual Tester ─────────────────────────────────────────────────────────────

const log = (msg, cls='ok') => {{
  const el = document.getElementById('console');
  el.className = 'console ' + cls;
  el.textContent = msg;
}};

async function doReset() {{
  log('Resetting environment...', 'loading');
  manualReward = 0; manualSteps = 0;
  updateScore(0, 0, {{}});
  try {{
    const r = await fetch('/reset', {{ method:'POST' }});
    const d = await r.json();
    log(JSON.stringify(d, null, 2), 'ok');
    document.getElementById('step-btn').disabled = false;
    document.getElementById('score-section').style.display = 'block';
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

async function doStep() {{
  log('Submitting mock action...', 'loading');
  const action = {{
    hallucination_detected: true,
    hallucination_explanation: "The AI analysis validates claims that are factually incorrect or unverifiable.",
    bias_detected: false,
    bias_explanation: "No significant systematic bias detected against any identifiable group.",
    alignment_violated: true,
    alignment_explanation: "The post violates platform rules regarding misinformation and unverified claims.",
    memory_consistent: true,
    memory_explanation: "The author's current post is consistent with patterns in their previous posts.",
    overall_verdict: "remove",
    confidence: 0.78
  }};
  try {{
    const r = await fetch('/step', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(action) }});
    const d = await r.json();
    manualReward += d.reward || 0;
    manualSteps += 1;
    updateScore(manualReward, manualSteps, d.info?.breakdown || {{}});
    log(JSON.stringify(d, null, 2), 'ok');
    if (d.done) {{
      document.getElementById('step-btn').disabled = true;
      document.getElementById('step-btn').textContent = '✓ Episode Complete';
    }}
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

async function doState() {{
  log('Fetching state...', 'loading');
  try {{
    const r = await fetch('/state');
    log(JSON.stringify(await r.json(), null, 2));
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

async function doHealth() {{
  log('Checking health...', 'loading');
  try {{
    const r = await fetch('/health');
    log(JSON.stringify(await r.json(), null, 2), 'ok');
  }} catch(e) {{ log('Error: ' + e.message, 'err'); }}
}}

function updateScore(reward, steps, bd) {{
  document.getElementById('score-num').textContent = reward.toFixed(3);
  document.getElementById('score-fill').style.width = Math.min((reward/5)*100, 100) + '%';
  document.getElementById('score-meta').textContent = steps + ' / 5 tasks · ' + (steps>0?(reward/steps).toFixed(3):'0.000') + ' avg/task';
  document.getElementById('score-section').style.display = 'block';
  const dims = [['hallucination','🤥 Hall.'],['bias','⚖️ Bias'],['alignment','📋 Align.'],['memory','🧠 Mem.'],['verdict','✅ Verdict']];
  document.getElementById('breakdown').innerHTML = dims.map(([k,l]) =>
    `<div class="rb-item"><div class="rb-val">${{bd[k]!==undefined?bd[k].toFixed(2):'—'}}</div><div class="rb-label">${{l}}</div></div>`
  ).join('');
}}
</script>
</body>
</html>""")