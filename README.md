---
title: Invoice Processing Agent
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - invoice-processing
  - agent
license: mit
---

# 🧾 Invoice Processing Agent — OpenEnv

An OpenEnv environment where an AI agent processes invoices and decides whether to **approve**, **reject**, or **flag** them for human review.

## Why This Problem?

Invoice processing is one of the most time-consuming tasks in any organization. Finance teams manually review hundreds of invoices weekly, checking for duplicates, anomalies, missing fields, and fraud signals. This environment simulates that workflow so AI agents can learn to automate it.

---

## Action Space

| Field      | Type   | Values                        |
|------------|--------|-------------------------------|
| `decision` | string | `approve`, `reject`, `flag`   |
| `reason`   | string | Short explanation (free text) |

## Observation Space

| Field         | Type       | Description                              |
|---------------|------------|------------------------------------------|
| `invoice_id`  | string     | Unique invoice identifier                |
| `vendor`      | string     | Vendor name (may be empty)               |
| `amount`      | float      | Invoice total                            |
| `currency`    | string     | Currency code (USD, EUR, etc.)           |
| `date`        | string     | Invoice date                             |
| `line_items`  | list[dict] | Individual line items                    |
| `flags`       | list[str]  | Detected anomalies                       |
| `done`        | bool       | Is the episode over?                     |
| `reward`      | float      | Reward for the last step                 |
| `message`     | string     | Human-readable feedback                  |

---

## Tasks

| Task            | Difficulty | # Invoices | Description                                     |
|-----------------|------------|------------|-------------------------------------------------|
| `easy_triage`   | Easy       | 3          | Clear approve/reject signals                    |
| `medium_triage` | Medium     | 4          | Duplicates, over-budget, multi-vendor           |
| `hard_triage`   | Hard       | 5          | Missing fields, currency mismatches, anomalies  |

---

## Reward Function

| Outcome              | Reward |
|----------------------|--------|
| Correct decision     | +1.0   |
| Cautious flag        | +0.3   |
| Wrong decision       | 0.0    |

---

## Baseline Scores

| Task            | Score |
|-----------------|-------|
| easy_triage     | 1.00  |
| medium_triage   | 0.75  |
| hard_triage     | 0.60  |

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
python -m uvicorn server.app:app --reload --port 7860
```

### Docker

```bash
docker build -t invoice-env .
docker run -p 7860:7860 invoice-env
```

### Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py
```

---

## API Endpoints

| Endpoint         | Method | Description                          |
|------------------|--------|--------------------------------------|
| `/reset`         | POST   | Start a new episode                  |
| `/step`          | POST   | Submit a decision                    |
| `/state`         | GET    | Get current episode state            |
| `/tasks`         | GET    | List tasks + action schema           |
| `/grader`        | POST   | Get final episode score              |
| `/baseline` | POST | Run rule-based agent, returns scores for all 3 tasks |
| `/health`        | GET    | Health check                         |

---

## Example Usage

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_name": "easy_triage"}).json()

while not obs["done"]:
    # Agent decides
    action = {"decision": "approve", "reason": "Looks valid"}
    result = requests.post(f"{BASE}/step", json=action).json()
    obs = result["observation"]
    print(obs["message"])

# Get final score
score = requests.post(f"{BASE}/grader").json()
print(f"Score: {score['score']}")
```
