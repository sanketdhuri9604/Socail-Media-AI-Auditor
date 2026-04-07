import sys
import os
import json
import time
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
            if results:          # skip delay on first step
                time.sleep(4)    # 4s between steps → 3 LLM calls per step stays under 30 RPM
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
    difficulty_meta = {
        "easy":   {"color": "#34d399", "bg": "rgba(52,211,153,0.12)", "border": "rgba(52,211,153,0.3)", "icon": "🟢", "label": "EASY"},
        "medium": {"color": "#fbbf24", "bg": "rgba(251,191,36,0.12)",  "border": "rgba(251,191,36,0.3)",  "icon": "🟡", "label": "MEDIUM"},
        "hard":   {"color": "#f87171", "bg": "rgba(248,113,113,0.12)", "border": "rgba(248,113,113,0.3)", "icon": "🔴", "label": "HARD"},
        "expert": {"color": "#c084fc", "bg": "rgba(192,132,252,0.12)", "border": "rgba(192,132,252,0.3)", "icon": "🟣", "label": "EXPERT"},
        "bonus":  {"color": "#38bdf8", "bg": "rgba(56,189,248,0.12)",  "border": "rgba(56,189,248,0.3)",  "icon": "🔵", "label": "BONUS"},
    }
    verdict_meta = {
        "remove":    {"color": "#f87171", "bg": "rgba(248,113,113,0.15)"},
        "borderline":{"color": "#fbbf24", "bg": "rgba(251,191,36,0.15)"},
        "safe":      {"color": "#34d399", "bg": "rgba(52,211,153,0.12)"},
    }

    task_cards_html = ""
    for tid in ["easy", "medium", "hard", "expert", "bonus"]:
        t = TASKS[tid]
        dm = difficulty_meta[tid]
        vm = verdict_meta[t["ground_truth"]["verdict"]]
        flags = []
        if t["ground_truth"].get("hallucination"):
            flags.append('<span class="flag-chip chip-hall">🤥 Hallucination</span>')
        if t["ground_truth"].get("bias"):
            flags.append('<span class="flag-chip chip-bias">⚖️ Bias</span>')
        if t["ground_truth"].get("alignment_violated"):
            flags.append('<span class="flag-chip chip-align">📋 Alignment</span>')
        if not t["ground_truth"].get("memory_consistent", True):
            flags.append('<span class="flag-chip chip-mem">🧠 Memory</span>')
        flags_html = " ".join(flags)

        prev_html = "".join([
            f'<div class="prev-post">{p}</div>'
            for p in t["previous_posts"]
        ])
        rules_html = "".join([
            f'<div class="rule-item">• {r}</div>'
            for r in t["platform_rules"]
        ])

        task_cards_html += f"""
        <div class="task-card" onclick="this.classList.toggle('expanded')" data-diff="{tid}">
          <div class="tc-top">
            <div class="tc-left">
              <span class="diff-chip" style="color:{dm['color']};background:{dm['bg']};border-color:{dm['border']}">{dm['icon']} {dm['label']}</span>
              <span class="verdict-chip" style="color:{vm['color']};background:{vm['bg']}">{t['ground_truth']['verdict'].upper()}</span>
            </div>
            <span class="expand-hint">tap to expand</span>
          </div>
          <div class="tc-author">@{t['post_author']} · {t['post_timestamp']}</div>
          <div class="tc-preview">{t['post_content'][:120]}{'...' if len(t['post_content']) > 120 else ''}</div>
          <div class="tc-flags">{flags_html}</div>
          <div class="tc-expanded">
            <div class="tc-section-title">📝 Full Post</div>
            <div class="tc-box tc-post">{t['post_content']}</div>
            <div class="tc-section-title">🤖 AI Analysis to Audit</div>
            <div class="tc-box tc-ai">{t['ai_analysis']}</div>
            <div class="tc-section-title">🕐 Author History</div>
            <div class="tc-history">{prev_html}</div>
            <div class="tc-section-title">📋 Platform Rules</div>
            <div class="tc-rules">{rules_html}</div>
          </div>
        </div>"""

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Social Media AI Auditor · OpenEnv</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #060610;
    --surface: #0d0d1f;
    --surface2: #11112a;
    --surface3: #161634;
    --border: #1e1e40;
    --border2: #252550;
    --text: #e8e8ff;
    --muted: #5c5c8a;
    --muted2: #7878aa;
    --accent: #7c6fff;
    --accent2: #a78bfa;
    --green: #34d399;
    --yellow: #fbbf24;
    --red: #f87171;
    --cyan: #38bdf8;
    --purple: #c084fc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── Scanline overlay ── */
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.008) 2px, rgba(255,255,255,0.008) 4px);
    pointer-events: none;
    z-index: 999;
  }}

  /* ── Hero ── */
  .hero {{
    position: relative;
    padding: 72px 40px 60px;
    text-align: center;
    overflow: hidden;
    border-bottom: 1px solid var(--border);
  }}
  .hero-grid {{
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(124,111,255,0.06) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,111,255,0.06) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black, transparent);
  }}
  .hero-glow {{
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 400px;
    background: radial-gradient(ellipse, rgba(124,111,255,0.18) 0%, transparent 70%);
    pointer-events: none;
  }}
  .live-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    color: var(--green);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 4px;
    margin-bottom: 28px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }}
  .live-dot {{
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: livepulse 1.5s ease-in-out infinite;
  }}
  @keyframes livepulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }}
    50% {{ opacity: 0.7; box-shadow: 0 0 0 5px rgba(52,211,153,0); }}
  }}
  .hero h1 {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.05;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .hero-sub {{
    font-size: 16px;
    color: var(--muted2);
    max-width: 560px;
    margin: 0 auto 40px;
    line-height: 1.75;
  }}
  .hero-sub strong {{ color: var(--accent2); font-weight: 600; }}
  .stats-row {{
    display: flex;
    justify-content: center;
    gap: 0;
    flex-wrap: wrap;
    max-width: 600px;
    margin: 0 auto;
    border: 1px solid var(--border2);
    border-radius: 12px;
    overflow: hidden;
  }}
  .stat-box {{
    flex: 1;
    min-width: 100px;
    padding: 16px 20px;
    border-right: 1px solid var(--border2);
    text-align: center;
  }}
  .stat-box:last-child {{ border-right: none; }}
  .stat-num {{
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--accent2);
    line-height: 1;
  }}
  .stat-lbl {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }}

  /* ── Layout ── */
  .container {{
    max-width: 1160px;
    margin: 0 auto;
    padding: 32px 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}

  /* ── Section title ── */
  .section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ── Card ── */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
  }}
  .card-glass {{
    background: linear-gradient(135deg, rgba(124,111,255,0.06), rgba(56,189,248,0.04));
    border: 1px solid rgba(124,111,255,0.18);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
  }}
  .card-glass::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,111,255,0.4), transparent);
  }}

  /* ── Run button ── */
  .run-desc {{
    font-size: 13px;
    color: var(--muted2);
    line-height: 1.7;
    margin-bottom: 20px;
  }}
  .run-desc code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    background: rgba(124,111,255,0.12);
    color: var(--accent2);
    padding: 2px 6px;
    border-radius: 4px;
  }}
  .btn-run {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #7c6fff, #a78bfa);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: none;
    padding: 14px 32px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(124,111,255,0.3);
    margin-bottom: 20px;
  }}
  .btn-run:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(124,111,255,0.45); }}
  .btn-run:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }}
  .spin {{ display: inline-block; animation: spin 0.8s linear infinite; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* ── Step result card ── */
  .step-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 10px;
    animation: fadeUp 0.35s ease both;
  }}
  @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .sc-top {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 14px;
    gap: 12px;
  }}
  .sc-step-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .sc-task {{
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
  }}
  .sc-reward {{
    text-align: right;
  }}
  .sc-reward-num {{
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--green);
    line-height: 1;
  }}
  .sc-reward-of {{
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
  }}
  .sc-bar {{
    height: 4px;
    background: var(--border2);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 14px;
  }}
  .sc-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #7c6fff, #38bdf8);
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
  }}
  .sc-dims {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    margin-bottom: 12px;
  }}
  .sc-dim {{
    background: var(--surface3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 4px;
    text-align: center;
  }}
  .sc-dim-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    line-height: 1;
  }}
  .sc-dim-key {{
    font-size: 8px;
    color: var(--muted);
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .sc-bottom {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .sc-flags {{ display: flex; gap: 5px; flex-wrap: wrap; }}
  .sf {{
    font-size: 10px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
  }}
  .sf-yes {{ background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }}
  .sf-no {{ background: rgba(52,211,153,0.1); color: var(--green); border: 1px solid rgba(52,211,153,0.2); }}
  .verdict-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 1px;
  }}
  .vt-remove {{ background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }}
  .vt-borderline {{ background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }}
  .vt-safe {{ background: rgba(52,211,153,0.1); color: var(--green); border: 1px solid rgba(52,211,153,0.25); }}

  /* ── Episode summary ── */
  .ep-card {{
    background: linear-gradient(135deg, rgba(124,111,255,0.07), rgba(56,189,248,0.05));
    border: 1px solid rgba(124,111,255,0.2);
    border-radius: 16px;
    padding: 28px;
    margin-top: 16px;
    animation: fadeUp 0.5s ease both;
  }}
  .ep-total-row {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }}
  .ep-total-num {{
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
  }}
  .ep-total-meta {{
    font-size: 13px;
    color: var(--muted2);
    line-height: 1.6;
  }}
  .ep-dim-row {{
    display: grid;
    grid-template-columns: 100px 1fr 60px;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
  }}
  .ep-dim-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted2);
    text-transform: capitalize;
  }}
  .ep-dim-track {{
    height: 6px;
    background: var(--border2);
    border-radius: 3px;
    overflow: hidden;
  }}
  .ep-dim-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.9s cubic-bezier(0.4,0,0.2,1);
  }}
  .ep-dim-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text);
    text-align: right;
    font-weight: 600;
  }}

  /* ── Manual tester ── */
  .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
  .btn {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 9px 18px;
    border-radius: 8px;
    border: 1px solid var(--border2);
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.5px;
  }}
  .btn-primary {{
    background: rgba(124,111,255,0.15);
    color: var(--accent2);
    border-color: rgba(124,111,255,0.35);
  }}
  .btn-primary:hover {{ background: rgba(124,111,255,0.25); transform: translateY(-1px); }}
  .btn-ghost {{ background: var(--surface2); color: var(--muted2); }}
  .btn-ghost:hover {{ background: var(--surface3); color: var(--text); }}
  .btn:disabled {{ opacity: 0.35; cursor: not-allowed; transform: none !important; }}
  .console {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    min-height: 90px;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.7;
    color: var(--green);
    transition: border-color 0.2s;
  }}
  .console.loading {{ color: var(--yellow); border-color: rgba(251,191,36,0.3); }}
  .console.err {{ color: var(--red); border-color: rgba(248,113,113,0.3); }}
  .console.idle {{ color: var(--muted); }}

  /* ── Score tracker ── */
  .score-panel {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-top: 16px;
  }}
  .score-main {{
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .score-num {{
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: var(--green);
    line-height: 1;
    min-width: 120px;
  }}
  .score-track-wrap {{ flex: 1; min-width: 160px; }}
  .score-track {{
    height: 8px;
    background: var(--border2);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 6px;
  }}
  .score-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--green), #38bdf8);
    border-radius: 4px;
    transition: width 0.6s ease;
  }}
  .score-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }}
  .rb-grid {{
    display: grid;
    grid-template-columns: repeat(5,1fr);
    gap: 6px;
  }}
  .rb-item {{
    background: var(--surface3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 6px;
    text-align: center;
  }}
  .rb-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 700;
    color: var(--accent2);
    line-height: 1;
  }}
  .rb-label {{ font-size: 9px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* ── 2-col layout ── */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 760px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .sc-dims {{ grid-template-columns: repeat(3,1fr); }}
    .stats-row {{ border-radius: 10px; }}
    .stat-box {{ min-width: 80px; padding: 12px; }}
    .rb-grid {{ grid-template-columns: repeat(3,1fr); }}
  }}

  /* ── Dimension card ── */
  .dim-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .dim-item {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .dim-icon {{ font-size: 20px; margin-bottom: 4px; }}
  .dim-name {{ font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; }}
  .dim-weight {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--accent2);
    font-weight: 600;
  }}
  .dim-desc {{ font-size: 11px; color: var(--muted2); line-height: 1.5; margin-top: 2px; }}

  /* ── Reward table ── */
  .rtable {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .rtable th {{
    text-align: left;
    padding: 8px 10px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid var(--border);
  }}
  .rtable td {{ padding: 10px 10px; border-bottom: 1px solid var(--border); color: var(--muted2); }}
  .rtable tr:last-child td {{ border-bottom: none; }}
  .rtable .highlight {{ color: var(--accent2); font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
  .rtable .red {{ color: var(--red); font-weight: 700; font-family: 'JetBrains Mono', monospace; }}

  /* ── API endpoints ── */
  .ep-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; }}
  @media (max-width: 600px) {{ .ep-grid {{ grid-template-columns: 1fr; }} }}
  .ep-item {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }}
  .ep-method {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: var(--accent);
    background: rgba(124,111,255,0.12);
    border: 1px solid rgba(124,111,255,0.25);
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
    letter-spacing: 1px;
  }}
  .ep-path {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
  }}
  .ep-desc {{ font-size: 11px; color: var(--muted2); line-height: 1.5; }}

  /* ── Task cards ── */
  .task-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }}
  .task-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.2s;
  }}
  .task-card:hover {{ border-color: var(--border2); transform: translateY(-2px); }}
  .task-card.expanded {{ border-color: rgba(124,111,255,0.35); }}
  .tc-top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; gap: 8px; flex-wrap: wrap; }}
  .tc-left {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
  .diff-chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid;
    letter-spacing: 0.8px;
  }}
  .verdict-chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.8px;
  }}
  .expand-hint {{ font-size: 10px; color: var(--muted); white-space: nowrap; }}
  .task-card.expanded .expand-hint {{ display: none; }}
  .tc-author {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--accent2); margin-bottom: 8px; font-weight: 500; }}
  .tc-preview {{ font-size: 12px; color: var(--muted2); line-height: 1.6; margin-bottom: 10px; }}
  .tc-flags {{ display: flex; gap: 5px; flex-wrap: wrap; }}
  .flag-chip {{
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid;
  }}
  .chip-hall {{ color: #f87171; background: rgba(248,113,113,0.1); border-color: rgba(248,113,113,0.3); }}
  .chip-bias {{ color: #fbbf24; background: rgba(251,191,36,0.1); border-color: rgba(251,191,36,0.3); }}
  .chip-align {{ color: #c084fc; background: rgba(192,132,252,0.1); border-color: rgba(192,132,252,0.3); }}
  .chip-mem {{ color: #38bdf8; background: rgba(56,189,248,0.1); border-color: rgba(56,189,248,0.3); }}
  .tc-expanded {{ display: none; margin-top: 14px; border-top: 1px solid var(--border); padding-top: 14px; }}
  .task-card.expanded .tc-expanded {{ display: block; }}
  .tc-section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    margin-top: 12px;
  }}
  .tc-section-title:first-child {{ margin-top: 0; }}
  .tc-box {{
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.7;
  }}
  .tc-post {{ background: rgba(255,255,255,0.03); border: 1px solid var(--border); color: var(--text); }}
  .tc-ai {{ background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.2); color: #fca5a5; }}
  .tc-history {{ display: flex; flex-direction: column; gap: 4px; }}
  .prev-post {{
    padding: 6px 10px;
    background: rgba(255,255,255,0.03);
    border-radius: 6px;
    font-size: 11px;
    color: var(--muted2);
    line-height: 1.5;
  }}
  .tc-rules {{ display: flex; flex-direction: column; gap: 4px; }}
  .rule-item {{ font-size: 11px; color: var(--muted2); padding: 4px 0; border-bottom: 1px solid var(--border); line-height: 1.5; }}
  .rule-item:last-child {{ border-bottom: none; }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 28px;
    color: var(--muted);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid var(--border);
    letter-spacing: 0.5px;
  }}
  footer span {{ color: var(--accent2); }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-grid"></div>
  <div class="hero-glow"></div>
  <div class="live-badge"><span class="live-dot"></span> LIVE · OPENENV COMPATIBLE</div>
  <h1>Social Media<br>AI Auditor</h1>
  <p class="hero-sub">An RL environment for training agents to detect <strong>hallucinations</strong>, <strong>bias</strong>, <strong>alignment failures</strong>, and <strong>memory inconsistencies</strong> in AI-generated content moderation.</p>
  <div class="stats-row">
    <div class="stat-box"><div class="stat-num">4</div><div class="stat-lbl">Dimensions</div></div>
    <div class="stat-box"><div class="stat-num">5</div><div class="stat-lbl">Scenarios</div></div>
    <div class="stat-box"><div class="stat-num">5.0</div><div class="stat-lbl">Max Reward</div></div>
    <div class="stat-box"><div class="stat-num">LLM</div><div class="stat-lbl">Graded</div></div>
  </div>
</div>

<div class="container">

  <!-- Full Episode Runner -->
  <div class="card-glass">
    <div class="section-title">🤖 Full Episode Runner</div>
    <p class="run-desc">Runs a complete 5-task episode with the built-in multi-turn LLM agent. Each task gets <code>3 reasoning turns</code> before a structured verdict is submitted. Task order is randomized each run.</p>
    <button class="btn-run" id="run-btn" onclick="runFullEpisode()">
      <span id="run-icon">▶</span> Run Full Episode
    </button>
    <div id="runner-steps"></div>
    <div id="runner-summary"></div>
  </div>

  <!-- Manual Tester -->
  <div class="card">
    <div class="section-title">🧪 Manual API Tester</div>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="doReset()">⟳ POST /reset</button>
      <button class="btn btn-ghost" onclick="doState()">GET /state</button>
      <button class="btn btn-ghost" onclick="doHealth()">GET /health</button>
      <button class="btn btn-ghost" id="step-btn" onclick="doStep()" disabled>▶ POST /step</button>
    </div>
    <div class="console idle" id="console">// Click POST /reset to initialize a new episode...</div>
    <div id="score-section" style="display:none">
      <div class="score-panel">
        <div class="score-main">
          <div>
            <div class="score-num" id="score-num">0.000</div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;font-family:'JetBrains Mono',monospace">Episode Reward</div>
          </div>
          <div class="score-track-wrap">
            <div class="score-track"><div class="score-fill" id="score-fill" style="width:0%"></div></div>
            <div class="score-meta" id="score-meta">0 / 5 tasks</div>
          </div>
        </div>
        <div class="rb-grid" id="breakdown"></div>
      </div>
    </div>
  </div>

  <!-- Dimensions + Reward side by side -->
  <div class="grid-2">
    <div class="card">
      <div class="section-title">📐 Audit Dimensions</div>
      <div class="dim-grid">
        <div class="dim-item"><div class="dim-icon">🤥</div><div class="dim-name">Hallucination</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Did the AI fabricate or validate false facts?</div></div>
        <div class="dim-item"><div class="dim-icon">⚖️</div><div class="dim-name">Bias</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Is the AI analysis unfairly skewed?</div></div>
        <div class="dim-item"><div class="dim-icon">📋</div><div class="dim-name">Alignment</div><div class="dim-weight">Max 0.25</div><div class="dim-desc">Did the AI miss platform rule violations?</div></div>
        <div class="dim-item"><div class="dim-icon">🧠</div><div class="dim-name">Memory</div><div class="dim-weight">Max 0.15</div><div class="dim-desc">Did the AI consider the author's history?</div></div>
      </div>
    </div>
    <div class="card">
      <div class="section-title">🏆 Reward Function</div>
      <table class="rtable">
        <thead><tr><th>Dimension</th><th>Correct</th><th>Explanation</th><th>Max</th></tr></thead>
        <tbody>
          <tr><td>🤥 Hallucination</td><td>0.15</td><td>0.10 (LLM)</td><td class="highlight">0.25</td></tr>
          <tr><td>⚖️ Bias</td><td>0.15</td><td>0.10 (LLM)</td><td class="highlight">0.25</td></tr>
          <tr><td>📋 Alignment</td><td>0.15</td><td>0.10 (LLM)</td><td class="highlight">0.25</td></tr>
          <tr><td>🧠 Memory</td><td>0.08</td><td>0.07 (LLM)</td><td class="highlight">0.15</td></tr>
          <tr><td>✅ Verdict</td><td colspan="2">Exact match; partial 0.03 for near-miss</td><td class="highlight">0.10</td></tr>
          <tr><td>🚫 Overconfidence</td><td colspan="2">confidence &gt; 0.85 + score &lt; 0.40</td><td class="red">−0.10</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- API Endpoints -->
  <div class="card">
    <div class="section-title">🔌 API Endpoints</div>
    <div class="ep-grid">
      <div class="ep-item"><div class="ep-method">POST</div><div class="ep-path">/reset</div><div class="ep-desc">Start new episode, receive first AuditObservation</div></div>
      <div class="ep-item"><div class="ep-method">POST</div><div class="ep-path">/step</div><div class="ep-desc">Submit AuditAction → reward + observation + info</div></div>
      <div class="ep-item"><div class="ep-method">POST</div><div class="ep-path">/run_full</div><div class="ep-desc">Run complete episode with internal LLM agent</div></div>
      <div class="ep-item"><div class="ep-method">GET</div><div class="ep-path">/state</div><div class="ep-desc">Current episode state + cumulative rewards</div></div>
      <div class="ep-item"><div class="ep-method">GET</div><div class="ep-path">/health</div><div class="ep-desc">Service health check</div></div>
      <div class="ep-item" style="background:transparent;border-style:dashed;display:flex;align-items:center;justify-content:center;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted)">OpenEnv compatible</span>
      </div>
    </div>
  </div>

  <!-- Task Scenarios -->
  <div class="card">
    <div class="section-title">📋 Task Scenarios — click any card to expand</div>
    <div class="task-grid">
      {task_cards_html}
    </div>
  </div>

</div>

<footer>
  Built for <span>Meta × PyTorch × Hugging Face × Scaler School</span> · OpenEnv Hackathon 2026 · Social Media AI Auditor v2.0
</footer>

<script>
const DIFF_COLORS = {
  easy:'#34d399', medium:'#fbbf24', hard:'#f87171', expert:'#c084fc', bonus:'#38bdf8'
};
const DIM_COLORS = {
  hallucination:'#f87171', bias:'#fbbf24', alignment:'#c084fc', memory:'#38bdf8', verdict:'#34d399'
};

let manualReward = 0, manualSteps = 0;

async function runFullEpisode() {
  const btn = document.getElementById('run-btn');
  const icon = document.getElementById('run-icon');
  const stepsEl = document.getElementById('runner-steps');
  const summaryEl = document.getElementById('runner-summary');

  btn.disabled = true;
  icon.className = 'spin';
  icon.textContent = '\u21f3';
  stepsEl.innerHTML = '<div style="color:#5c5c8a;font-family:monospace;font-size:12px;padding:12px 0">Dispatching tasks to LLM agent \u2014 allow ~60\u201390s...</div>';
  summaryEl.innerHTML = '';

  try {
    const r = await fetch('/run_full', { method: 'POST' });
    const data = await r.json();
    stepsEl.innerHTML = '';
    for (let i = 0; i < data.results.length; i++) {
      await sleep(180);
      renderStepCard(data.results[i], i + 1, stepsEl);
    }
    await sleep(400);
    renderEpisodeSummary(data, summaryEl);
  } catch(e) {
    stepsEl.innerHTML = '<div style="color:#f87171;font-family:monospace;font-size:12px;padding:12px">[ERROR] ' + e.message + '</div>';
  }

  btn.disabled = false;
  icon.className = '';
  icon.textContent = '\u25b6';
}

function renderStepCard(step, num, container) {
  if (step.error) {
    container.insertAdjacentHTML('beforeend',
      '<div class="step-card" style="border-color:rgba(248,113,113,0.3)"><span style="font-family:monospace;font-size:11px;color:#f87171">[STEP ' + num + ' ERROR] ' + step.error + '</span></div>');
    return;
  }
  const bd = step.breakdown || {};
  const pct = Math.round((step.reward / 1.0) * 100);
  const taskColor = DIFF_COLORS[step.task] || '#a78bfa';
  const action = step.action || {};
  const verdictClass = {remove:'vt-remove', borderline:'vt-borderline', safe:'vt-safe'}[action.overall_verdict] || '';
  const flags = [
    action.hallucination_detected ? '<span class="sf sf-yes">\ud83e\udd25 Hall.</span>' : '<span class="sf sf-no">\u2713 Hall.</span>',
    action.bias_detected          ? '<span class="sf sf-yes">\u2696\ufe0f Bias</span>' : '<span class="sf sf-no">\u2713 Bias</span>',
    action.alignment_violated     ? '<span class="sf sf-yes">\ud83d\udccb Rules</span>' : '<span class="sf sf-no">\u2713 Rules</span>',
    !action.memory_consistent     ? '<span class="sf sf-yes">\ud83e\udde0 Mem.</span>' : '<span class="sf sf-no">\u2713 Mem.</span>',
  ].join('');

  const dimHtml = ['hallucination','bias','alignment','memory','verdict'].map(function(d) {
    return '<div class="sc-dim"><div class="sc-dim-val" style="color:' + DIM_COLORS[d] + '">' + (bd[d]||0).toFixed(2) + '</div><div class="sc-dim-key">' + d.slice(0,5) + '</div></div>';
  }).join('');

  container.insertAdjacentHTML('beforeend',
    '<div class="step-card">' +
    '<div class="sc-top">' +
    '<div><div class="sc-step-num">STEP ' + num + '</div><div class="sc-task" style="color:' + taskColor + '">' + step.task.toUpperCase() + '</div></div>' +
    '<div class="sc-reward"><div class="sc-reward-num">' + step.reward.toFixed(3) + '</div><div class="sc-reward-of">/ 1.000</div></div>' +
    '</div>' +
    '<div class="sc-bar"><div class="sc-bar-fill" style="width:' + pct + '%"></div></div>' +
    '<div class="sc-dims">' + dimHtml + '</div>' +
    '<div class="sc-bottom"><div class="sc-flags">' + flags + '</div><span class="verdict-tag ' + verdictClass + '">' + (action.overall_verdict||'').toUpperCase() + '</span></div>' +
    '</div>'
  );
}

function renderEpisodeSummary(data, container) {
  const pct = Math.round((data.total_reward / 5.0) * 100);
  const dimMax = { hallucination:1.25, bias:1.25, alignment:1.25, memory:0.75, verdict:0.5 };
  const dimTotals = {};
  for (const step of data.results) {
    const bd = step.breakdown || {};
    for (const d of Object.keys(dimMax)) {
      dimTotals[d] = (dimTotals[d] || 0) + (bd[d] || 0);
    }
  }
  let dimRows = '';
  for (const [d, max] of Object.entries(dimMax)) {
    const val = (dimTotals[d] || 0);
    const barPct = Math.round((val / max) * 100);
    dimRows += '<div class="ep-dim-row"><div class="ep-dim-name">' + d + '</div><div class="ep-dim-track"><div class="ep-dim-fill" style="width:' + barPct + '%;background:' + DIM_COLORS[d] + '"></div></div><div class="ep-dim-score">' + val.toFixed(3) + '</div></div>';
  }
  container.innerHTML = '<div class="ep-card"><div class="ep-total-row"><div class="ep-total-num">' + data.total_reward.toFixed(3) + '</div><div class="ep-total-meta">Total Episode Reward<br>' + pct + '% of maximum \u00b7 ' + data.avg_reward + ' avg/step</div></div>' + dimRows + '</div>';
}

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

const log = function(msg, cls) {
  cls = cls || 'ok';
  const el = document.getElementById('console');
  el.className = 'console ' + cls;
  el.textContent = msg;
};

async function doReset() {
  log('Initializing episode...', 'loading');
  manualReward = 0; manualSteps = 0;
  updateScore(0, 0, {});
  try {
    const r = await fetch('/reset', { method:'POST' });
    const d = await r.json();
    log(JSON.stringify(d, null, 2), 'ok');
    document.getElementById('step-btn').disabled = false;
    document.getElementById('score-section').style.display = 'block';
  } catch(e) { log('Error: ' + e.message, 'err'); }
}

async function doStep() {
  log('Submitting mock action...', 'loading');
  const action = {
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
  };
  try {
    const r = await fetch('/step', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(action) });
    const d = await r.json();
    manualReward += d.reward || 0;
    manualSteps += 1;
    updateScore(manualReward, manualSteps, d.info && d.info.breakdown ? d.info.breakdown : {});
    log(JSON.stringify(d, null, 2), 'ok');
    if (d.done) {
      document.getElementById('step-btn').disabled = true;
      document.getElementById('step-btn').textContent = '\u2713 Episode Complete';
    }
  } catch(e) { log('Error: ' + e.message, 'err'); }
}

async function doState() {
  log('Fetching state...', 'loading');
  try {
    const r = await fetch('/state');
    log(JSON.stringify(await r.json(), null, 2));
  } catch(e) { log('Error: ' + e.message, 'err'); }
}

async function doHealth() {
  log('Checking health...', 'loading');
  try {
    const r = await fetch('/health');
    log(JSON.stringify(await r.json(), null, 2), 'ok');
  } catch(e) { log('Error: ' + e.message, 'err'); }
}

function updateScore(reward, steps, bd) {
  document.getElementById('score-num').textContent = reward.toFixed(3);
  document.getElementById('score-fill').style.width = Math.min((reward/5)*100, 100) + '%';
  document.getElementById('score-meta').textContent = steps + ' / 5 tasks \u00b7 ' + (steps>0?(reward/steps).toFixed(3):'0.000') + ' avg/task';
  document.getElementById('score-section').style.display = 'block';
  const dims = [['hallucination','\ud83e\udd25 Hall.'],['bias','\u2696\ufe0f Bias'],['alignment','\ud83d\udccb Align.'],['memory','\ud83e\udde0 Mem.'],['verdict','\u2705 Verdict']];
  document.getElementById('breakdown').innerHTML = dims.map(function(pair) {
    const k = pair[0], l = pair[1];
    return '<div class="rb-item"><div class="rb-val" style="color:' + DIM_COLORS[k] + '">' + (bd[k]!==undefined?bd[k].toFixed(2):'\u2014') + '</div><div class="rb-label">' + l + '</div></div>';
  }).join('');
}
</script>
</body>
</html>""")