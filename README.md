---
title: Social Media AI Auditor Env
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Social Media AI Auditor — OpenEnv RL Environment
 
An RL environment that trains agents to detect **4 critical AI failure modes** in social media content analysis — simultaneously.
 
## Problem
 
AI systems analyzing social media can fail in 4 ways:
 
| Failure | What happens |
|---------|-------------|
| 🤥 Hallucination | AI makes up facts that don't exist |
| ⚖️ Bias | AI analysis unfairly targets specific groups |
| 📋 Alignment | AI misses platform rule violations |
| 🧠 Memory | AI ignores the author's posting history |
 
This environment trains agents to catch **all 4 failures at once** — exactly the challenge Meta faces at scale on Facebook and Instagram.
 
## Action Space
 
```python
AuditAction(
    hallucination_detected: bool,
    hallucination_explanation: str,
    bias_detected: bool,
    bias_explanation: str,
    alignment_violated: bool,
    alignment_explanation: str,
    memory_consistent: bool,
    memory_explanation: str,
    overall_verdict: "safe" | "borderline" | "remove",
    confidence: float  # 0.0 to 1.0
)
```
 
## Observation Space
 
```python
AuditObservation(
    post_content: str,
    post_author: str,
    post_timestamp: str,
    previous_posts: list[str],  # author history
    ai_analysis: str,           # flawed AI analysis to audit
    platform_rules: list[str],
    task_id: str,
    difficulty: str             # easy | medium | hard
)
```
 
## Reward Function
 
| Dimension | Max | Grading |
|-----------|-----|---------|
| Hallucination | 0.25 | 0.15 correct + 0.10 LLM explanation |
| Bias | 0.25 | 0.15 correct + 0.10 LLM explanation |
| Alignment | 0.25 | 0.15 correct + 0.10 LLM explanation |
| Memory | 0.15 | 0.08 correct + 0.07 LLM explanation |
| Verdict | 0.10 | Exact match |
| Overconfidence | -0.10 | Penalty if confident + wrong |
 
## Tasks
 
**Easy** — Obvious medical misinformation. AI analysis validates a false Harvard study claim.
 
**Medium** — Subtle regional bias. Real statistics framed divisively about North vs South India. AI misses the bias.
 
**Hard** — Coordinated fake whistleblowing. Author has a pattern of escalating unverifiable "insider" claims. AI incorrectly validates the source.
 
## API Endpoints
 
```
POST /reset   → Start new episode
POST /step    → Submit audit action
GET  /state   → Get current episode state
GET  /health  → Health check
GET  /        → Interactive web UI
```
 
## Setup
 
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
 
## Environment Variables
 
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key |
| `ENV_BASE_URL` | Running environment URL (for inference.py) |
 
## Run Inference
 
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_key"
export ENV_BASE_URL="http://localhost:8000"
 
python inference.py
```