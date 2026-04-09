# Social Media AI Auditor Env (Rebuilt Baseline)

This repository has been rebuilt from scratch as a contract-first OpenEnv project
to maximize compatibility with hackathon validators.

## What This Build Optimizes

- Exactly 3 tasks with explicit grader wiring.
- Task scores always in strict open interval (0, 1).
- Deterministic grading and deterministic inference output.
- Guaranteed START/STEP/END markers in inference logs.
- Minimal, stable payload schema for validator parsing.

## Project Structure

```text
app.py
inference.py
models.py
openenv.yaml
validate_presubmission.py
Dockerfile
pyproject.toml
requirements.txt
server/
  app.py
  environment.py
  grader.py
  tasks.py
  Dockerfile
```

## API Endpoints

- POST /reset
- POST /step
- GET /state
- GET /health
- POST /run_full
- GET /

The evaluator uses /reset and /step; /run_full is a helper endpoint for manual smoke tests.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

In another terminal:

```bash
python inference.py
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| API_BASE_URL | https://api.groq.com/openai/v1 | Required by hackathon checks |
| MODEL_NAME | llama-3.3-70b-versatile | Required by hackathon checks |
| API_KEY | empty | Optional secret source |
| HF_TOKEN | empty | Optional secret source |
| OPENAI_API_KEY | empty | Optional secret source |
| ENV_BASE_URL | http://localhost:7860 | Inference target server |
| ENV_RESET_MAX_ATTEMPTS | 6 | Retry count for /reset |
| ENV_RESET_RETRY_GAP | 2 | Delay between reset retries |
| STEP_DELAY | 0.5 | Delay between step calls |
| MINIMAL_END_PAYLOAD | 1 | Emit strict minimal END payload |

## Pre-Submission Validation

```bash
python validate_presubmission.py
```

The validator checks:

- required files and markers
- model typing
- grader output range
- non-constant grader behavior
- runtime END payload content from an actual inference run

## Docker

```bash
docker build -t social-media-auditor .
docker run -p 7860:7860 social-media-auditor
```

## Why This Rebuild Is Different

This implementation removes unnecessary runtime complexity and keeps the evaluator
surface area small and explicit. The design goal is predictable validator behavior,
not feature breadth.