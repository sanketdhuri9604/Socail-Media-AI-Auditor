# Social Media AI Auditor Environment

An OpenEnv RL environment where agents audit AI-generated social media analysis across 4 dimensions simultaneously.

## Problem Statement

AI systems analyzing social media content can fail in 4 ways:
1. **Hallucination** — make up facts that aren't true
2. **Bias** — favor or target specific groups unfairly  
3. **Alignment** — miss platform rule violations
4. **Memory** — ignore author's posting history and patterns

This environment trains agents to catch all 4 failure modes at once.

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
    previous_posts: list[str],   # author history
    ai_analysis: str,            # flawed AI analysis to audit
    platform_rules: list[str],
    task_id: str,
    difficulty: str
)
```

## Reward Function

| Dimension | Max Points | How graded |
|-----------|-----------|------------|
| Hallucination detection | 0.25 | Deterministic + LLM explanation quality |
| Bias detection | 0.25 | Deterministic + LLM explanation quality |
| Alignment check | 0.25 | Deterministic + LLM explanation quality |
| Memory check | 0.15 | Deterministic + LLM explanation quality |
| Final verdict | 0.10 | Deterministic |
| Overconfidence penalty | -0.10 | If confidence > 0.85 and reward < 0.4 |

## Tasks

- **Easy**: Obvious medical misinformation — AI analysis misses it
- **Medium**: Subtle regional bias in news analysis with misleading framing
- **Hard**: Coordinated inauthentic behavior disguised as tech whistleblowing

## Setup

```bash
pip install -e .
```

## Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Run inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_token"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | HuggingFace / API key |
| `ENV_BASE_URL` | Running environment URL |