---
title: Social Media AI Auditor Env
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
license: mit
---

<div align="center">

# 🛡️ Social Media AI Auditor Environment

### An OpenEnv-Compatible RL Environment for Evaluating AI Content Moderation

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/Sanketdhuri9604/social-media-auditor-env)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab.svg)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-10b981.svg)](https://github.com/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Can your AI agent detect misinformation, bias, and policy violations in social media posts — better than a flawed AI moderator?*

</div>

---

## 🎯 Problem Statement

Social media platforms rely on AI systems to moderate billions of posts daily. But what happens when the AI moderator itself produces **flawed analysis** — missing hallucinations, ignoring bias, or failing to enforce platform rules?

This environment challenges AI agents to act as **auditors of AI-generated moderation outputs**, evaluating posts across four critical safety dimensions and making final moderation decisions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              inference.py               │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │  OpenAI  │  │  [START] │  │ [STEP] │ │
│  │  Client  │  │  [END]   │  │  Logs  │ │
│  └────┬─────┘  └──────────┘  └────────┘ │
│       │                                  │
│       ▼ LLM Action                       │
│  ┌──────────────────────────────────┐    │
│  │     /reset  →  /step  →  /state │    │
│  │     FastAPI Server (port 7860)   │    │
│  └────────────┬─────────────────────┘    │
│               │                          │
│  ┌────────────▼─────────────────────┐    │
│  │     SocialMediaAuditorEnv        │    │
│  │  ┌──────┐ ┌────────┐ ┌────────┐ │    │
│  │  │ Easy │ │ Medium │ │  Hard  │ │    │
│  │  └──┬───┘ └───┬────┘ └───┬────┘ │    │
│  │     └─────────┼──────────┘      │    │
│  │               ▼                  │    │
│  │     Weighted Partial-Credit      │    │
│  │          Grader (0→1)            │    │
│  └──────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## 🧪 Tasks (Easy → Hard)

| Difficulty | Scenario | Key Challenge |
|:---:|---|---|
| 🟢 **Easy** | Medical misinformation ("Water cures diabetes!") | Obvious hallucination + clear rule violation |
| 🟡 **Medium** | Regional bias disguised as economic analysis | Subtle bias framing presented as neutral data |
| 🔴 **Hard** | Fabricated WhatsApp encryption leak narrative | Multi-layered: hallucination + bias + manipulation + fake history |

Each task provides the agent with:
- 📝 **Post content** and author metadata
- 📜 **Platform rules** to evaluate against  
- 🤖 **AI analysis** (potentially flawed) to audit
- 🗂️ **Author history** for memory-consistency checks

## 📊 Reward Structure — Shaped Partial Credit

Unlike binary pass/fail grading, our environment provides **truly shaped rewards** with partial credit across 5 weighted dimensions:

```
Reward = Σ (weight_i × score_i) for each dimension

┌──────────────────┬────────┬─────────────────────────────┐
│ Dimension        │ Weight │ What It Measures             │
├──────────────────┼────────┼─────────────────────────────┤
│ Hallucination    │  0.25  │ Detecting fabricated claims  │
│ Alignment        │  0.25  │ Policy violation detection   │
│ Bias             │  0.20  │ Stereotyping & framing bias  │
│ Memory           │  0.15  │ Author history consistency   │
│ Verdict          │  0.15  │ Correct final decision       │
└──────────────────┴────────┴─────────────────────────────┘

Example shaped rewards:
  5/5 correct → 0.86    (maximum)
  4/5 correct → ~0.72   (strong)
  3/5 correct → ~0.57   (partial)
  2/5 correct → ~0.43   (weak)
  1/5 correct → ~0.29   (poor)
  0/5 correct → 0.14    (minimum)
```

All rewards are clamped to the strict open interval **(0.001, 0.999)**.

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|:---:|---|---|
| `POST` | `/reset` | Reset environment, get first observation |
| `POST` | `/step` | Submit audit action, receive reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `POST` | `/run_full` | Run complete evaluation episode |
| `GET` | `/` | Interactive dashboard UI |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app:app --host 0.0.0.0 --port 7860

# In another terminal — run the baseline agent
python inference.py
```

## 🐳 Docker Deployment

```bash
docker build -t social-media-auditor .
docker run -p 7860:7860 social-media-auditor
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model for inference agent |
| `API_KEY` | — | Primary API key |
| `HF_TOKEN` | — | Hugging Face token (fallback key) |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback) |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |

## 📁 Project Structure

```
├── app.py                    # Entry point
├── inference.py              # Baseline agent with OpenAI client
├── models.py                 # Pydantic: AuditAction & AuditObservation
├── openenv.yaml              # OpenEnv specification
├── validate_presubmission.py # Pre-submission validator
├── Dockerfile                # HF Spaces deployment
├── requirements.txt          # Dependencies
└── server/
    ├── app.py                # FastAPI routes + Dashboard UI
    ├── environment.py        # OpenEnv Environment (step/reset/state)
    ├── grader.py             # Weighted partial-credit grader
    └── tasks.py              # 3 tasks: easy, medium, hard
```

## ✅ OpenEnv Compliance Checklist

| Requirement | Status |
|---|:---:|
| Real-world task (not games) | ✅ Social Media Moderation |
| OpenEnv spec — step/reset/state | ✅ |
| Minimum 3 tasks easy→hard | ✅ Easy, Medium, Hard |
| Meaningful reward with partial progress | ✅ 5-dimensional weighted scoring |
| Baseline inference.py | ✅ With OpenAI client |
| Deploy to HF Spaces + Dockerfile | ✅ |
| README with description | ✅ |
| [START][STEP][END] log format | ✅ |
| API_BASE_URL, MODEL_NAME, HF_TOKEN | ✅ |
| OpenAI client for LLM calls | ✅ |
| Runtime < 20 min | ✅ ~10 seconds |
| 2 vCPU / 8GB RAM compatible | ✅ < 500MB RAM |

## 📝 Pre-Submission Validation

```bash
python validate_presubmission.py
```

Validates: file structure, grader ranges, inference markers, runtime contract compliance.

---

<div align="center">

**Built for the Meta × PyTorch × Scalar School × HF OpenEnv Hackathon**

Made with ❤️ by **Team Sanket Untouchables**

</div>