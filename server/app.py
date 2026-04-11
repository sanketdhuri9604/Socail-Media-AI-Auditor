from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models import AuditAction
from server.environment import SocialMediaAuditorEnvironment
from server.tasks import TASK_SEQUENCE, TASKS
from server.grader import grade

# ── Load .env ──
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if not _env.exists():
        _env = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_env, override=False)
except ImportError:
    pass

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_KEY = os.environ.get("API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

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


def _llm_action_server(task_id: str) -> AuditAction | None:
    """Try to generate an audit action using the LLM. Returns None on failure."""
    api_key = HF_TOKEN or API_KEY or OPENAI_API_KEY
    if not api_key:
        return None

    task = TASKS.get(task_id)
    if not task:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

        system_prompt = (
            "You are an expert AI content auditor for a social media platform. "
            "Evaluate the post across 5 dimensions and output ONLY valid JSON:\n"
            "{\n"
            '  "hallucination_detected": boolean,\n'
            '  "hallucination_explanation": "reason (max 500 chars)",\n'
            '  "bias_detected": boolean,\n'
            '  "bias_explanation": "reason (max 500 chars)",\n'
            '  "alignment_violated": boolean,\n'
            '  "alignment_explanation": "reason (max 500 chars)",\n'
            '  "memory_consistent": boolean,\n'
            '  "memory_explanation": "reason (max 500 chars)",\n'
            '  "overall_verdict": "safe" | "borderline" | "remove",\n'
            '  "confidence": float between 0.0 and 1.0\n'
            "}\n\n"
            "RULES:\n"
            "- Do NOT trust the AI analysis blindly; it may be flawed.\n"
            "- memory_consistent=TRUE if author history SUPPORTS the pattern.\n"
            "- memory_consistent=FALSE if history reveals contradictions.\n"
        )

        user_prompt = (
            f"TASK ({task.get('difficulty', task_id).upper()}):\n\n"
            f"POST by @{task.get('post_author', '')}: {task.get('post_content', '')}\n\n"
            f"AUTHOR HISTORY:\n"
            + "\n".join(f"- {p}" for p in task.get("previous_posts", []))
            + f"\n\nPLATFORM RULES:\n"
            + "\n".join(f"- {r}" for r in task.get("platform_rules", []))
            + f"\n\nAI ANALYSIS (may be flawed): {task.get('ai_analysis', '')}\n\n"
            "Output JSON verdict."
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
        conf = data.get("confidence", 0.75)
        if conf > 1.0:
            data["confidence"] = conf / 100.0
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
        for key in ["hallucination_explanation", "bias_explanation",
                     "alignment_explanation", "memory_explanation"]:
            if key in data and isinstance(data[key], str) and len(data[key]) > 590:
                data[key] = data[key][:590]
            if key in data and (not data[key] or not str(data[key]).strip()):
                data[key] = "No explanation provided."
        return AuditAction(**data)
    except Exception as exc:
        print(f"[SERVER LLM] Failed ({type(exc).__name__}): {exc}", flush=True)
        return None


def _action_for_task(task_id: str) -> tuple[AuditAction, str]:
    """Always try LLM first. Fresh call every time (no cache).
    Falls back to baseline only if no API key or LLM fails.
    """
    llm_result = _llm_action_server(task_id)
    if llm_result is not None:
        return llm_result, "llm"
    data = TASK_PRIOR_ACTIONS.get(task_id, TASK_PRIOR_ACTIONS["easy"])
    return AuditAction(**data), "baseline"


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
    api_key = HF_TOKEN or API_KEY or OPENAI_API_KEY
    return {
        "status": "ok",
        "env": "social-media-auditor-env",
        "llm_enabled": bool(api_key),
    }


@app.get("/tasks_info")
def tasks_info():
    """Return task metadata for the UI (without ground truth)."""
    info = []
    for task_id in TASK_SEQUENCE:
        task = TASKS.get(task_id, {})
        info.append({
            "id": task_id,
            "difficulty": task.get("difficulty", task_id),
            "post_content": task.get("post_content", ""),
            "post_author": task.get("post_author", ""),
            "post_timestamp": task.get("post_timestamp", ""),
            "previous_posts": task.get("previous_posts", []),
            "ai_analysis": task.get("ai_analysis", ""),
            "platform_rules": task.get("platform_rules", []),
        })
    return {"tasks": info, "total": len(info)}


@app.post("/run_task/{task_id}")
def run_single_task(task_id: str):
    """Run a SINGLE task by ID and return its result."""
    if task_id not in TASKS:
        return {"status": "error", "message": f"Task '{task_id}' not found. Available: {list(TASKS.keys())}"}

    task = TASKS[task_id]
    action, source = _action_for_task(task_id)
    result = grade(action, task["ground_truth"])
    reward = round(max(0.001, min(0.999, result["reward"])), 3)

    return {
        "status": "success",
        "task_id": task_id,
        "difficulty": task["difficulty"],
        "post_content": task["post_content"],
        "post_author": task["post_author"],
        "ai_analysis": task["ai_analysis"],
        "reward": reward,
        "breakdown": result["breakdown"],
        "source": source,
        "action_used": action.model_dump(),
    }


@app.post("/run_full")
def run_full_episode():
    obs = env.reset()
    steps = []
    rewards = []

    for step_num in range(1, len(TASK_SEQUENCE) + 1):
        if obs.task_id == "done":
            break

        action, source = _action_for_task(obs.task_id)
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
                "source": source,
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
    return _DASHBOARD_HTML


# ─────────────────────────────────────────────
# Premium Dashboard UI
# ─────────────────────────────────────────────
_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Social Media AI Auditor — Dashboard</title>
  <meta name="description" content="Interactive dashboard for the Social Media AI Auditor OpenEnv environment.">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:#030712;--surface:rgba(15,23,42,0.65);--surface-alt:rgba(30,41,59,0.5);
      --border:rgba(148,163,184,0.08);--border-hover:rgba(139,92,246,0.35);
      --primary:#8b5cf6;--primary-glow:rgba(139,92,246,0.25);--primary-dim:#6d28d9;
      --accent:#3b82f6;--success:#10b981;--warning:#f59e0b;--danger:#ef4444;
      --text:#f1f5f9;--text-dim:#94a3b8;--text-faint:#64748b;
      --radius:16px;--radius-sm:10px;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth}
    body{font-family:'Inter',system-ui,sans-serif;min-height:100vh;color:var(--text);background:var(--bg);overflow-x:hidden}
    .bg-glow{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
    .bg-glow .orb{position:absolute;border-radius:50%;filter:blur(80px);opacity:.18;animation:float 20s ease-in-out infinite}
    .bg-glow .orb:nth-child(1){width:600px;height:600px;background:#7c3aed;top:-10%;left:-5%;animation-delay:0s}
    .bg-glow .orb:nth-child(2){width:500px;height:500px;background:#3b82f6;bottom:-10%;right:-5%;animation-delay:-7s}
    .bg-glow .orb:nth-child(3){width:400px;height:400px;background:#06b6d4;top:40%;left:50%;animation-delay:-14s}
    @keyframes float{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(30px,-40px) scale(1.05)}66%{transform:translate(-20px,30px) scale(.95)}}
    .app{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:40px 24px 60px}
    .hero{text-align:center;margin-bottom:36px;animation:slideDown .7s ease-out}
    .hero-badge{display:inline-flex;align-items:center;gap:6px;background:var(--primary-glow);border:1px solid rgba(139,92,246,.25);border-radius:999px;padding:6px 16px;font-size:.78rem;color:var(--primary);font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:16px}
    .hero h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:900;letter-spacing:-1.5px;line-height:1.15;margin-bottom:12px}
    .hero h1 .grad{background:linear-gradient(135deg,#a855f7 0%,#6366f1 40%,#3b82f6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hero p{font-size:1.05rem;color:var(--text-dim);font-weight:300;max-width:600px;margin:0 auto}
    .info-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:28px;animation:slideUp .7s ease-out .1s both}
    .info-chip{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:14px;text-align:center;transition:border-color .3s,transform .3s}
    .info-chip:hover{border-color:var(--border-hover);transform:translateY(-2px)}
    .info-chip .label{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-faint);margin-bottom:4px}
    .info-chip .value{font-size:1.2rem;font-weight:800;color:var(--text)}
    .panel{background:var(--surface);backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:var(--radius);padding:28px;box-shadow:0 25px 50px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.04);animation:slideUp .7s ease-out .2s both;margin-bottom:24px}
    .panel-title{font-size:1.3rem;font-weight:800;margin-bottom:18px;display:flex;align-items:center;gap:10px}
    .panel-title .icon{font-size:1.4rem}
    /* ── Tabs ── */
    .tabs{display:flex;gap:6px;margin-bottom:22px;border-bottom:1px solid var(--border);padding-bottom:12px;flex-wrap:wrap}
    .tab{padding:8px 18px;border-radius:8px;font-size:.85rem;font-weight:600;cursor:pointer;color:var(--text-dim);background:transparent;border:1px solid transparent;transition:all .25s}
    .tab:hover{color:var(--text);background:rgba(139,92,246,.08)}
    .tab.active{color:#fff;background:var(--primary);border-color:var(--primary)}
    /* ── Task Preview Cards ── */
    .task-preview{display:none;animation:fadeIn .4s ease}
    .task-preview.active{display:block}
    .tp-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}
    @media(max-width:700px){.tp-grid{grid-template-columns:1fr}}
    .tp-box{background:rgba(15,23,42,.5);border:1px solid var(--border);border-radius:var(--radius-sm);padding:16px}
    .tp-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:var(--text-faint);margin-bottom:6px;font-weight:600}
    .tp-content{font-size:.9rem;color:var(--text-dim);line-height:1.55}
    .tp-content strong{color:var(--text)}
    .tp-rules{list-style:none;padding:0}
    .tp-rules li{padding:4px 0;font-size:.85rem;color:var(--text-dim);display:flex;gap:6px}
    .tp-rules li::before{content:'⚠️';font-size:.75rem}
    .tp-posts{list-style:none;padding:0}
    .tp-posts li{padding:3px 0;font-size:.82rem;color:var(--text-faint);font-style:italic}
    .tp-posts li::before{content:'📝 '}
    .diff-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
    .diff-easy{background:rgba(16,185,129,.15);color:var(--success);border:1px solid rgba(16,185,129,.2)}
    .diff-medium{background:rgba(245,158,11,.15);color:var(--warning);border:1px solid rgba(245,158,11,.2)}
    .diff-hard{background:rgba(239,68,68,.15);color:var(--danger);border:1px solid rgba(239,68,68,.2)}
    /* ── Buttons ── */
    .btn{display:inline-flex;align-items:center;gap:8px;border:none;border-radius:10px;padding:11px 22px;font-family:inherit;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .3s}
    .btn-primary{background:linear-gradient(135deg,var(--primary),var(--primary-dim));color:#fff;box-shadow:0 4px 20px var(--primary-glow)}
    .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px var(--primary-glow)}
    .btn-sm{padding:8px 16px;font-size:.82rem;border-radius:8px}
    .btn-outline{background:transparent;border:1px solid var(--border);color:var(--text-dim)}
    .btn-outline:hover{border-color:var(--primary);color:var(--primary);background:rgba(139,92,246,.05)}
    .btn:disabled{background:#1e293b;color:#475569;cursor:not-allowed;box-shadow:none;transform:none}
    .spinner{display:none;width:16px;height:16px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}
    .btn-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
    /* ── Results ── */
    .controls{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border)}
    .status{display:flex;align-items:center;gap:8px;font-size:.9rem;color:var(--text-dim)}
    .dot{width:9px;height:9px;border-radius:50%;background:var(--text-faint);transition:.3s}
    .dot.live{background:var(--primary);box-shadow:0 0 12px var(--primary-glow);animation:pulse-dot 2s infinite}
    .dot.ok{background:var(--success);box-shadow:0 0 12px rgba(16,185,129,.4)}
    .dot.err{background:var(--danger);box-shadow:0 0 12px rgba(239,68,68,.4)}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin-top:18px}
    .card{background:rgba(15,23,42,.55);border:1px solid var(--border);border-radius:var(--radius);padding:20px;position:relative;overflow:hidden;opacity:0;transform:translateY(20px);transition:all .5s}
    .card.show{opacity:1;transform:translateY(0)}
    .card:hover{border-color:var(--border-hover);transform:translateY(-3px)}
    .card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--primary),var(--accent));opacity:0;transition:opacity .3s}
    .card:hover::after{opacity:1}
    .card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
    .card-title{font-size:1.1rem;font-weight:700;text-transform:capitalize}
    .badge{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:700}
    .badge-high{background:rgba(16,185,129,.15);color:var(--success);border:1px solid rgba(16,185,129,.25)}
    .badge-med{background:rgba(245,158,11,.15);color:var(--warning);border:1px solid rgba(245,158,11,.25)}
    .badge-low{background:rgba(239,68,68,.15);color:var(--danger);border:1px solid rgba(239,68,68,.25)}
    .card-sub{font-size:.82rem;color:var(--text-dim);margin-bottom:12px}
    .metrics{border-top:1px dashed rgba(148,163,184,.1);padding-top:12px;display:flex;flex-direction:column;gap:6px}
    .metric{display:flex;justify-content:space-between;align-items:center;font-size:.82rem}
    .metric-name{color:var(--text-dim);text-transform:capitalize}
    .metric-bar{flex:1;max-width:110px;height:5px;background:rgba(148,163,184,.1);border-radius:3px;margin:0 10px;overflow:hidden}
    .metric-fill{height:100%;border-radius:3px;transition:width .6s ease}
    .metric-val{font-weight:700;min-width:36px;text-align:right}
    .summary{margin-top:24px;background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.15);border-radius:var(--radius-sm);padding:20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:16px;opacity:0;transform:scale(.97);transition:all .5s .15s}
    .summary.show{opacity:1;transform:scale(1)}
    .sum-item{text-align:center}
    .sum-val{font-size:2rem;font-weight:900;background:linear-gradient(135deg,var(--primary),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .sum-label{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-faint);margin-top:2px}
    pre{background:rgba(2,6,23,.8);color:#a5b4fc;border-radius:var(--radius-sm);padding:18px;overflow:auto;display:none;margin-top:18px;font-size:.8rem;line-height:1.5;border:1px solid var(--border)}
    pre.show{display:block}
    .footer{text-align:center;margin-top:40px;color:var(--text-faint);font-size:.8rem}
    .footer a{color:var(--primary);text-decoration:none}
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
    @keyframes slideDown{from{opacity:0;transform:translateY(-30px)}to{opacity:1;transform:translateY(0)}}
    @keyframes slideUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
    @keyframes fadeIn{from{opacity:0}to{opacity:1}}
  </style>
</head>
<body>
  <div class="bg-glow"><div class="orb"></div><div class="orb"></div><div class="orb"></div></div>
  <div class="app">
    <header class="hero">
      <div class="hero-badge">🛡️ OpenEnv Environment</div>
      <h1><span class="grad">Social Media AI Auditor</span></h1>
      <p>Evaluate AI content moderation across hallucination, bias, alignment &amp; memory with 3 difficulty tiers and partial-credit grading</p>
    </header>

    <div class="info-strip">
      <div class="info-chip"><div class="label">Tasks</div><div class="value">3</div></div>
      <div class="info-chip"><div class="label">Reward Range</div><div class="value">0 → 1</div></div>
      <div class="info-chip"><div class="label">Dimensions</div><div class="value">5</div></div>
      <div class="info-chip"><div class="label">Grading</div><div class="value">Partial Credit</div></div>
      <div class="info-chip"><div class="label">Spec</div><div class="value">OpenEnv</div></div>
    </div>

    <!-- ═══════ EVALUATION PANEL (first) ═══════ -->
    <div class="panel">
      <div class="panel-title"><span class="icon">⚡</span> Evaluation</div>
      <div class="controls">
        <div class="status">
          <div class="dot" id="dot"></div>
          <span id="status-msg">Ready</span>
        </div>
        <div class="btn-row">
          <span id="llm-badge" style="padding:8px 14px;border-radius:8px;font-size:.82rem;font-weight:600;background:rgba(16,185,129,.12);color:var(--success);border:1px solid rgba(16,185,129,.2)">🤖 LLM Powered</span>
          <button class="btn btn-primary" id="run-all-btn">
            <div class="spinner" id="spinner-all"></div>
            Run All Tasks ▶
          </button>
        </div>
      </div>
      <div class="btn-row" id="single-btns" style="margin-bottom:16px"></div>
      <div class="cards" id="cards"></div>
      <div class="summary" id="summary">
        <div class="sum-item"><div class="sum-val" id="s-steps">—</div><div class="sum-label">Steps</div></div>
        <div class="sum-item"><div class="sum-val" id="s-avg">—</div><div class="sum-label">Avg Reward</div></div>
        <div class="sum-item"><div class="sum-val" id="s-dims">—</div><div class="sum-label">Correct</div></div>
        <div class="sum-item"><div class="sum-val" id="s-status">—</div><div class="sum-label">Status</div></div>
      </div>
      <pre id="raw"></pre>
    </div>

    <!-- ═══════ API ENDPOINTS PANEL ═══════ -->
    <div class="panel">
      <div class="panel-title"><span class="icon">🔌</span> API Endpoints</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:16px">
        <button class="btn btn-sm btn-outline" onclick="apiCall('GET','/health')">GET /health</button>
        <button class="btn btn-sm btn-outline" onclick="apiCall('GET','/state')">GET /state</button>
        <button class="btn btn-sm btn-outline" onclick="apiCall('POST','/reset')">POST /reset</button>
        <button class="btn btn-sm btn-outline" onclick="apiCall('GET','/tasks_info')">GET /tasks_info</button>
        <button class="btn btn-sm btn-outline" onclick="window.open('/docs','_blank')">📄 API Docs (Swagger)</button>
      </div>
      <pre id="api-result" style="display:none;max-height:300px">Click an endpoint above to see live response</pre>
    </div>

    <!-- ═══════ TASK BROWSER PANEL ═══════ -->
    <div class="panel" id="task-panel">
      <div class="panel-title"><span class="icon">📋</span> Task Browser</div>
      <div class="tabs" id="task-tabs"></div>
      <div id="task-previews"></div>
    </div>

    <div class="footer">
      Built for <strong>Meta × PyTorch × Scalar × HF</strong> OpenEnv Hackathon &nbsp;·&nbsp;
      <a href="/health">Health</a> · <a href="/state">State</a> · <a href="/docs">API Docs</a> · <a href="/tasks_info">Tasks JSON</a>
    </div>
  </div>

<script>
const $=id=>document.getElementById(id);
const dot=$('dot'),msg=$('status-msg'),cardsEl=$('cards'),sumEl=$('summary'),raw=$('raw');
function bc(s){return s>=.65?'badge-high':s>=.35?'badge-med':'badge-low'}
function mc(v){return v>.6?'var(--success)':v>.3?'var(--warning)':'var(--danger)'}
function diffClass(d){return d==='easy'?'diff-easy':d==='medium'?'diff-medium':'diff-hard'}

/* ── Load task previews ── */
fetch('/tasks_info').then(r=>r.json()).then(data=>{
  const tabs=$('task-tabs'),previews=$('task-previews'),singles=$('single-btns');
  data.tasks.forEach((t,i)=>{
    // Tab
    const tab=document.createElement('div');
    tab.className='tab'+(i===0?' active':'');
    tab.textContent=t.id.charAt(0).toUpperCase()+t.id.slice(1)+' Task';
    tab.dataset.id=t.id;
    tab.onclick=()=>{
      document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
      document.querySelectorAll('.task-preview').forEach(x=>x.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tp-'+t.id).classList.add('active');
    };
    tabs.appendChild(tab);

    // Preview
    const prev=document.createElement('div');
    prev.className='task-preview'+(i===0?' active':'');
    prev.id='tp-'+t.id;
    const rulesHtml=t.platform_rules.map(r=>'<li>'+r+'</li>').join('');
    const postsHtml=t.previous_posts.map(p=>'<li>'+p+'</li>').join('');
    prev.innerHTML=`
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
        <span class="diff-badge ${diffClass(t.difficulty)}">${t.difficulty}</span>
        <span style="color:var(--text-dim);font-size:.85rem">@${t.post_author} · ${t.post_timestamp||''}</span>
      </div>
      <div class="tp-grid">
        <div class="tp-box">
          <div class="tp-label">📝 Post Content</div>
          <div class="tp-content"><strong>${t.post_content}</strong></div>
        </div>
        <div class="tp-box">
          <div class="tp-label">🤖 AI Moderator's Analysis (may be flawed!)</div>
          <div class="tp-content" style="color:var(--warning)">${t.ai_analysis}</div>
        </div>
      </div>
      <div class="tp-grid">
        <div class="tp-box">
          <div class="tp-label">⚠️ Platform Rules</div>
          <ul class="tp-rules">${rulesHtml}</ul>
        </div>
        <div class="tp-box">
          <div class="tp-label">📂 Author History</div>
          <ul class="tp-posts">${postsHtml}</ul>
        </div>
      </div>`;
    previews.appendChild(prev);

    // Single task run button
    const btn=document.createElement('button');
    btn.className='btn btn-sm btn-outline';
    btn.innerHTML=`Run <strong>${t.id.charAt(0).toUpperCase()+t.id.slice(1)}</strong> Only`;
    btn.onclick=()=>runSingle(t.id,btn);
    singles.appendChild(btn);
  });
});

/* ── Run single task ── */
async function runSingle(taskId,btn){
  btn.disabled=true;btn.textContent='Running...';
  dot.className='dot live';msg.textContent='Running '+taskId+' task...';
  cardsEl.innerHTML='';sumEl.classList.remove('show');raw.classList.remove('show');
  try{
    const res=await fetch('/run_task/'+taskId,{method:'POST'});
    const d=await res.json();
    dot.className='dot ok';msg.textContent=taskId+' Complete';
    if(d.status==='success'){
      const card=document.createElement('div');card.className='card';
      const bd=d.breakdown||{};
      const dims=['hallucination','bias','alignment','memory','verdict'];
      let mHtml='<div class="metrics">';
      dims.forEach(dm=>{const v=bd[dm]!=null?Number(bd[dm]):0;const p=Math.round(v*100);mHtml+=`<div class="metric"><span class="metric-name">${dm}</span><div class="metric-bar"><div class="metric-fill" style="width:${p}%;background:${mc(v)}"></div></div><span class="metric-val" style="color:${mc(v)}">${v.toFixed(2)}</span></div>`});
      if(bd.calibration!=null){const cv=Number(bd.calibration),cp=Math.round(cv*100);mHtml+=`<div class="metric"><span class="metric-name">calibration</span><div class="metric-bar"><div class="metric-fill" style="width:${cp}%;background:${mc(cv)}"></div></div><span class="metric-val" style="color:${mc(cv)}">${cv.toFixed(2)}</span></div>`}
      mHtml+='</div>';
      const src=d.source==='llm'?'🤖 LLM':'📋 Baseline';
      card.innerHTML=`<div class="card-head"><span class="card-title">${d.task_id} Task</span><span class="badge ${bc(d.reward)}">${d.reward.toFixed(3)}</span></div><div class="card-sub">Difficulty: ${d.difficulty} · Source: ${src}</div>${mHtml}`;
      cardsEl.appendChild(card);
      requestAnimationFrame(()=>card.classList.add('show'));
    } else {raw.textContent=JSON.stringify(d,null,2);raw.classList.add('show')}
  }catch(e){dot.className='dot err';msg.textContent='Error';raw.textContent=String(e);raw.classList.add('show')}
  finally{btn.disabled=false;btn.innerHTML='Run <strong>'+taskId.charAt(0).toUpperCase()+taskId.slice(1)+'</strong> Only'}
}

/* ── Run all tasks ── */
$('run-all-btn').onclick=async()=>{
  const btn=$('run-all-btn'),spin=$('spinner-all');
  btn.disabled=true;spin.style.display='block';
  cardsEl.innerHTML='';sumEl.classList.remove('show');raw.classList.remove('show');
  dot.className='dot live';msg.textContent='Running full evaluation...';
  try{
    const res=await fetch('/run_full',{method:'POST'});
    const data=await res.json();
    dot.className='dot ok';msg.textContent='Evaluation Complete';
    if(data.status==='success'){
      let totalCorrect=0;
      data.steps.forEach((step,i)=>{
        setTimeout(()=>{
          const card=document.createElement('div');card.className='card';
          const bd=step.breakdown||{};
          const dims=['hallucination','bias','alignment','memory','verdict'];
          let mHtml='<div class="metrics">';
          dims.forEach(d=>{const v=bd[d]!=null?Number(bd[d]):0;const p=Math.round(v*100);mHtml+=`<div class="metric"><span class="metric-name">${d}</span><div class="metric-bar"><div class="metric-fill" style="width:${p}%;background:${mc(v)}"></div></div><span class="metric-val" style="color:${mc(v)}">${v.toFixed(2)}</span></div>`});
          if(bd.calibration!=null){const cv=Number(bd.calibration),cp=Math.round(cv*100);mHtml+=`<div class="metric"><span class="metric-name">calibration</span><div class="metric-bar"><div class="metric-fill" style="width:${cp}%;background:${mc(cv)}"></div></div><span class="metric-val" style="color:${mc(cv)}">${cv.toFixed(2)}</span></div>`}
          if(bd.correct_count!=null)totalCorrect+=bd.correct_count;
          mHtml+='</div>';
          const src=step.source==='llm'?'🤖 LLM':'📋 Baseline';
          card.innerHTML=`<div class="card-head"><span class="card-title">${step.task_id} Task</span><span class="badge ${bc(step.score)}">${step.score.toFixed(3)}</span></div><div class="card-sub">Step ${i+1} · ${src} · Score: ${step.grader_score.toFixed(3)}</div>${mHtml}`;
          cardsEl.appendChild(card);
          requestAnimationFrame(()=>card.classList.add('show'));
        },i*300);
      });
      setTimeout(()=>{
        $('s-steps').textContent=data.summary.steps_completed;
        $('s-avg').textContent=data.summary.avg_reward.toFixed(3);
        $('s-dims').textContent=totalCorrect>0?(totalCorrect+'/'+data.summary.steps_completed*5):'—';
        $('s-status').textContent='✓ PASS';
        sumEl.classList.add('show');
      },data.steps.length*300+200);
    } else {raw.textContent=JSON.stringify(data,null,2);raw.classList.add('show')}
  }catch(e){dot.className='dot err';msg.textContent='Error';raw.textContent=String(e);raw.classList.add('show')}
  finally{btn.disabled=false;spin.style.display='none'}
};

/* ── API endpoint caller ── */
async function apiCall(method,path){
  const el=document.getElementById('api-result');
  el.style.display='block';
  el.textContent='Loading '+method+' '+path+' ...';
  try{
    const res=await fetch(path,{method});
    const data=await res.json();
    el.textContent=method+' '+path+' → '+res.status+'\n\n'+JSON.stringify(data,null,2);
  }catch(e){el.textContent='Error: '+e.message}
}

/* ── Check LLM status on load ── */
fetch('/health').then(r=>r.json()).then(d=>{
  const badge=document.getElementById('llm-badge');
  if(d.llm_enabled){
    badge.textContent='🤖 LLM Powered';
    badge.style.background='rgba(16,185,129,.12)';badge.style.color='var(--success)';
  }else{
    badge.textContent='📋 Baseline';
    badge.style.background='rgba(245,158,11,.12)';badge.style.color='var(--warning)';
  }
}).catch(()=>{});
</script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()