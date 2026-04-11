from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models import AuditAction
from server.environment import SocialMediaAuditorEnvironment
from server.tasks import TASK_SEQUENCE, TASKS

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
            "platform_rules": task.get("platform_rules", []),
            "ai_analysis": task.get("ai_analysis", ""),
        })
    return {"tasks": info, "total": len(info)}


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
  <meta name="description" content="Interactive dashboard for the Social Media AI Auditor OpenEnv environment. Evaluate AI content moderation across hallucination, bias, alignment, and memory dimensions.">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #030712; --surface: rgba(15,23,42,0.65); --surface-alt: rgba(30,41,59,0.5);
      --border: rgba(148,163,184,0.08); --border-hover: rgba(139,92,246,0.35);
      --primary: #8b5cf6; --primary-glow: rgba(139,92,246,0.25); --primary-dim: #6d28d9;
      --accent: #3b82f6; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
      --text: #f1f5f9; --text-dim: #94a3b8; --text-faint: #64748b;
      --radius: 16px; --radius-sm: 10px;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth}
    body{
      font-family:'Inter',system-ui,sans-serif; min-height:100vh; color:var(--text);
      background:var(--bg); overflow-x:hidden;
    }

    /* ── Animated background ── */
    .bg-glow{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
    .bg-glow .orb{position:absolute;border-radius:50%;filter:blur(80px);opacity:.18;animation:float 20s ease-in-out infinite}
    .bg-glow .orb:nth-child(1){width:600px;height:600px;background:#7c3aed;top:-10%;left:-5%;animation-delay:0s}
    .bg-glow .orb:nth-child(2){width:500px;height:500px;background:#3b82f6;bottom:-10%;right:-5%;animation-delay:-7s}
    .bg-glow .orb:nth-child(3){width:400px;height:400px;background:#06b6d4;top:40%;left:50%;animation-delay:-14s}
    @keyframes float{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(30px,-40px) scale(1.05)}66%{transform:translate(-20px,30px) scale(.95)}}

    /* ── Layout ── */
    .app{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:40px 24px 60px}

    /* ── Header ── */
    .hero{text-align:center;margin-bottom:48px;animation:slideDown .7s ease-out}
    .hero-badge{display:inline-flex;align-items:center;gap:6px;background:var(--primary-glow);border:1px solid rgba(139,92,246,.25);border-radius:999px;padding:6px 16px;font-size:.78rem;color:var(--primary);font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:16px}
    .hero h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:900;letter-spacing:-1.5px;line-height:1.15;margin-bottom:12px}
    .hero h1 .grad{background:linear-gradient(135deg,#a855f7 0%,#6366f1 40%,#3b82f6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hero p{font-size:1.05rem;color:var(--text-dim);font-weight:300;max-width:600px;margin:0 auto}

    /* ── Info strip ── */
    .info-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:32px;animation:slideUp .7s ease-out .1s both}
    .info-chip{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:16px;text-align:center;transition:border-color .3s,transform .3s}
    .info-chip:hover{border-color:var(--border-hover);transform:translateY(-2px)}
    .info-chip .label{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-faint);margin-bottom:6px}
    .info-chip .value{font-size:1.3rem;font-weight:800;color:var(--text)}

    /* ── Main panel ── */
    .panel{background:var(--surface);backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:var(--radius);padding:32px;box-shadow:0 25px 50px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.04);animation:slideUp .7s ease-out .2s both}

    .controls{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid var(--border)}
    .status{display:flex;align-items:center;gap:10px;font-size:.95rem;color:var(--text-dim)}
    .dot{width:10px;height:10px;border-radius:50%;background:var(--text-faint);transition:.3s}
    .dot.live{background:var(--primary);box-shadow:0 0 12px var(--primary-glow);animation:pulse-dot 2s infinite}
    .dot.ok{background:var(--success);box-shadow:0 0 12px rgba(16,185,129,.4)}
    .dot.err{background:var(--danger);box-shadow:0 0 12px rgba(239,68,68,.4)}

    /* ── Buttons ── */
    .btn{display:inline-flex;align-items:center;gap:10px;border:none;border-radius:12px;padding:13px 26px;font-family:inherit;font-size:1rem;font-weight:600;cursor:pointer;transition:all .3s}
    .btn-primary{background:linear-gradient(135deg,var(--primary),var(--primary-dim));color:#fff;box-shadow:0 4px 20px var(--primary-glow)}
    .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 30px var(--primary-glow)}
    .btn:disabled{background:#1e293b;color:#475569;cursor:not-allowed;box-shadow:none;transform:none}
    .spinner{display:none;width:18px;height:18px;border:2.5px solid rgba(255,255,255,.25);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}

    /* ── Progress timeline ── */
    .timeline{display:flex;align-items:center;justify-content:center;gap:0;margin:24px 0 28px;opacity:0;transition:opacity .4s}
    .timeline.show{opacity:1}
    .tl-step{display:flex;flex-direction:column;align-items:center;position:relative;min-width:100px}
    .tl-circle{width:38px;height:38px;border-radius:50%;border:2px solid var(--border);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem;color:var(--text-faint);background:var(--bg);transition:all .4s;z-index:1}
    .tl-step.active .tl-circle{border-color:var(--primary);color:#fff;background:var(--primary);box-shadow:0 0 20px var(--primary-glow)}
    .tl-step.done .tl-circle{border-color:var(--success);color:#fff;background:var(--success)}
    .tl-label{margin-top:8px;font-size:.7rem;text-transform:uppercase;letter-spacing:.8px;color:var(--text-faint);font-weight:600}
    .tl-step.active .tl-label{color:var(--primary)}
    .tl-step.done .tl-label{color:var(--success)}
    .tl-line{width:60px;height:2px;background:var(--border);margin-bottom:20px;transition:background .4s}
    .tl-line.done{background:var(--success)}

    /* ── Task cards ── */
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:20px;margin-top:24px}
    .card{background:rgba(15,23,42,.55);border:1px solid var(--border);border-radius:var(--radius);padding:22px;position:relative;overflow:hidden;opacity:0;transform:translateY(24px);transition:all .5s ease}
    .card.show{opacity:1;transform:translateY(0)}
    .card:hover{border-color:var(--border-hover);transform:translateY(-4px)}
    .card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--primary),var(--accent));opacity:0;transition:opacity .3s}
    .card:hover::after{opacity:1}
    .card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
    .card-title{font-size:1.15rem;font-weight:700;text-transform:capitalize}
    .badge{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:700}
    .badge-high{background:rgba(16,185,129,.15);color:var(--success);border:1px solid rgba(16,185,129,.25)}
    .badge-med{background:rgba(245,158,11,.15);color:var(--warning);border:1px solid rgba(245,158,11,.25)}
    .badge-low{background:rgba(239,68,68,.15);color:var(--danger);border:1px solid rgba(239,68,68,.25)}
    .card-sub{font-size:.85rem;color:var(--text-dim);margin-bottom:14px}
    .metrics{border-top:1px dashed rgba(148,163,184,.1);padding-top:14px;display:flex;flex-direction:column;gap:8px}
    .metric{display:flex;justify-content:space-between;align-items:center;font-size:.85rem}
    .metric-name{color:var(--text-dim);text-transform:capitalize}
    .metric-bar{flex:1;max-width:120px;height:6px;background:rgba(148,163,184,.1);border-radius:3px;margin:0 12px;overflow:hidden}
    .metric-fill{height:100%;border-radius:3px;transition:width .6s ease}
    .metric-val{font-weight:700;min-width:40px;text-align:right}

    /* ── Summary ── */
    .summary{margin-top:28px;background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.15);border-radius:var(--radius-sm);padding:24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:20px;opacity:0;transform:scale(.97);transition:all .5s .15s}
    .summary.show{opacity:1;transform:scale(1)}
    .sum-item{text-align:center}
    .sum-val{font-size:2.2rem;font-weight:900;background:linear-gradient(135deg,var(--primary),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .sum-label{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-faint);margin-top:4px}

    pre{background:rgba(2,6,23,.8);color:#a5b4fc;border-radius:var(--radius-sm);padding:20px;overflow:auto;display:none;margin-top:20px;font-size:.82rem;line-height:1.6;border:1px solid var(--border)}
    pre.show{display:block}

    /* ── Footer ── */
    .footer{text-align:center;margin-top:48px;color:var(--text-faint);font-size:.82rem;animation:slideUp .7s ease-out .4s both}
    .footer a{color:var(--primary);text-decoration:none}

    /* ── Animations ── */
    @keyframes spin{to{transform:rotate(360deg)}}
    @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
    @keyframes slideDown{from{opacity:0;transform:translateY(-30px)}to{opacity:1;transform:translateY(0)}}
    @keyframes slideUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
  </style>
</head>
<body>
  <div class="bg-glow"><div class="orb"></div><div class="orb"></div><div class="orb"></div></div>

  <div class="app">
    <!-- Hero -->
    <header class="hero">
      <div class="hero-badge">🛡️ OpenEnv Environment</div>
      <h1><span class="grad">Auditor Analytics</span></h1>
      <p>Autonomous AI Content Moderation Evaluation — detect hallucinations, bias, and policy violations across 3 difficulty tiers</p>
    </header>

    <!-- Info strip -->
    <div class="info-strip">
      <div class="info-chip"><div class="label">Tasks</div><div class="value">3</div></div>
      <div class="info-chip"><div class="label">Reward Range</div><div class="value">0 → 1</div></div>
      <div class="info-chip"><div class="label">Dimensions</div><div class="value">5</div></div>
      <div class="info-chip"><div class="label">Grading</div><div class="value">Partial</div></div>
    </div>

    <!-- Main panel -->
    <div class="panel">
      <div class="controls">
        <div class="status">
          <div class="dot" id="dot"></div>
          <span id="status-msg">System Ready</span>
        </div>
        <button class="btn btn-primary" id="run-btn">
          <div class="spinner" id="spinner"></div>
          <span id="btn-label">Run Full Evaluation</span>
          <svg id="btn-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        </button>
      </div>

      <!-- Timeline -->
      <div class="timeline" id="timeline">
        <div class="tl-step" id="tl-1"><div class="tl-circle">1</div><div class="tl-label">Easy</div></div>
        <div class="tl-line" id="tl-l1"></div>
        <div class="tl-step" id="tl-2"><div class="tl-circle">2</div><div class="tl-label">Medium</div></div>
        <div class="tl-line" id="tl-l2"></div>
        <div class="tl-step" id="tl-3"><div class="tl-circle">3</div><div class="tl-label">Hard</div></div>
      </div>

      <!-- Task cards -->
      <div class="cards" id="cards"></div>

      <!-- Summary -->
      <div class="summary" id="summary">
        <div class="sum-item"><div class="sum-val" id="s-steps">—</div><div class="sum-label">Steps Completed</div></div>
        <div class="sum-item"><div class="sum-val" id="s-avg">—</div><div class="sum-label">Avg Reward</div></div>
        <div class="sum-item"><div class="sum-val" id="s-dims">—</div><div class="sum-label">Correct Dims</div></div>
        <div class="sum-item"><div class="sum-val" id="s-status">—</div><div class="sum-label">Status</div></div>
      </div>

      <pre id="raw"></pre>
    </div>

    <div class="footer">
      Built for <strong>Meta × PyTorch × Scalar × HF</strong> OpenEnv Hackathon &nbsp;·&nbsp;
      <a href="/health">Health</a> &nbsp;·&nbsp;
      <a href="/state">State</a> &nbsp;·&nbsp;
      <a href="/docs">API Docs</a>
    </div>
  </div>

<script>
const $ = id => document.getElementById(id);
const dot=$('dot'), msg=$('status-msg'), btn=$('run-btn'), spin=$('spinner'), lbl=$('btn-label'), ico=$('btn-icon'), cards=$('cards'), sum=$('summary'), raw=$('raw'), tl=$('timeline');

function badgeClass(s){return s>=.65?'badge-high':s>=.35?'badge-med':'badge-low';}
function barColor(v){return v>.6?'var(--success)':v>.3?'var(--warning)':'var(--danger)';}
function setTL(step,state){const el=$('tl-'+step);if(el)el.className='tl-step '+state;if(step>1){const ln=$('tl-l'+(step-1));if(ln&&state==='done')ln.className='tl-line done';}}

btn.onclick=async()=>{
  btn.disabled=true;spin.style.display='block';lbl.textContent='Processing…';ico.style.display='none';
  raw.classList.remove('show');cards.innerHTML='';sum.classList.remove('show');
  tl.classList.add('show');
  [1,2,3].forEach(i=>{setTL(i,'');const ln=$('tl-l'+i);if(ln)ln.className='tl-line';});
  dot.className='dot live';msg.textContent='Resetting environment…';

  try{
    const res=await fetch('/run_full',{method:'POST'});
    const data=await res.json();
    dot.className='dot ok';msg.textContent='Evaluation Complete';

    if(data.status==='success'){
      let totalCorrect=0;
      data.steps.forEach((step,i)=>{
        const n=i+1;
        setTL(n,'active');
        setTimeout(()=>{
          setTL(n,'done');
          const card=document.createElement('div');card.className='card';
          const bd=step.breakdown||{};
          const dims=['hallucination','bias','alignment','memory','verdict'];
          let metricsHtml='<div class="metrics">';
          dims.forEach(d=>{
            const v=bd[d]!=null?Number(bd[d]):0;
            const pct=Math.round(v*100);
            metricsHtml+=`<div class="metric"><span class="metric-name">${d}</span><div class="metric-bar"><div class="metric-fill" style="width:${pct}%;background:${barColor(v)}"></div></div><span class="metric-val" style="color:${barColor(v)}">${v.toFixed(2)}</span></div>`;
          });
          if(bd.calibration!=null){
            const cv=Number(bd.calibration),cpct=Math.round(cv*100);
            metricsHtml+=`<div class="metric"><span class="metric-name">calibration</span><div class="metric-bar"><div class="metric-fill" style="width:${cpct}%;background:${barColor(cv)}"></div></div><span class="metric-val" style="color:${barColor(cv)}">${cv.toFixed(2)}</span></div>`;
          }
          if(bd.correct_count!=null)totalCorrect+=bd.correct_count;
          metricsHtml+='</div>';

          card.innerHTML=`<div class="card-head"><span class="card-title">${step.task_id} Task</span><span class="badge ${badgeClass(step.score)}">${step.score.toFixed(3)}</span></div><div class="card-sub">Step ${n} · Grader: ${step.grader_score.toFixed(3)}</div>${metricsHtml}`;
          cards.appendChild(card);
          requestAnimationFrame(()=>card.classList.add('show'));
        },i*350);
      });

      setTimeout(()=>{
        $('s-steps').textContent=data.summary.steps_completed;
        $('s-avg').textContent=data.summary.avg_reward.toFixed(3);
        $('s-dims').textContent=totalCorrect+'/'+data.summary.steps_completed*5;
        $('s-status').textContent='✓ PASS';
        sum.classList.add('show');
      },data.steps.length*350+200);

    }else{raw.textContent=JSON.stringify(data,null,2);raw.classList.add('show');}
  }catch(e){
    dot.className='dot err';msg.textContent='Error: '+e.message;
    raw.textContent=String(e);raw.classList.add('show');
  }finally{
    btn.disabled=false;spin.style.display='none';lbl.textContent='Run Full Evaluation';ico.style.display='';
  }
};
</script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()