"""
Inference Script — Invoice Processing Agent
===================================
MANDATORY variables (set in environment):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face API key.
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")   # No default — must be set by user
ENV_URL      = os.getenv("ENV_URL", "https://sagar-03-invoice-processing-agent.hf.space")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# ── OpenAI client pointed at HuggingFace router ────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert invoice processing agent working for a finance team.

You will receive invoice details and must decide what to do with each invoice.

Your decision must be exactly one of:
- approve  → invoice looks valid, amounts match, vendor is known, no anomalies
- reject   → invoice is fraudulent, duplicate, from unknown/missing vendor, or clearly wrong
- flag     → invoice has anomalies needing human review (missing fields, currency mismatch, over budget)

Rules:
- Unknown or empty vendor name → reject
- Duplicate invoice (same vendor + amount + date) → reject
- Amount > $20,000 with vague description → reject
- Missing fields (empty vendor, no date) → reject
- Currency mismatch or unexpected currency → flag
- Amount unusually high for the vendor type → flag
- Line items don't add up to total → flag
- Everything looks clean and within normal range → approve

Respond ONLY with a valid JSON object, nothing else. No markdown, no explanation:
{"decision": "approve", "reason": "Brief reason here"}"""


def call_llm(invoice_obs: dict) -> dict:
    """Call the LLM with invoice details, return parsed action."""
    prompt = f"""Process this invoice and make a decision:

Invoice ID  : {invoice_obs['invoice_id']}
Vendor      : {invoice_obs['vendor'] or 'MISSING'}
Amount      : {invoice_obs['amount']} {invoice_obs['currency']}
Date        : {invoice_obs['date']}
Line Items  : {json.dumps(invoice_obs['line_items'], indent=2)}
Flags       : {invoice_obs['flags'] if invoice_obs['flags'] else 'None detected'}

Reply with JSON only: {{"decision": "approve/reject/flag", "reason": "brief reason"}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"decision": "flag", "reason": "Could not parse model response"}


def run_task(task_name: str) -> float:
    """Run the LLM agent on one task. Returns final score (0.0-1.0)."""

    # START log (required format)
    print(f"[START] {task_name}")

    resp = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": 42})
    resp.raise_for_status()
    obs = resp.json()

    total_reward = 0.0
    step = 0

    while not obs.get("done", False):
        step += 1
        action = call_llm(obs)
        decision = action.get("decision", "flag").lower().strip()
        reason   = action.get("reason", "")

        resp = requests.post(f"{ENV_URL}/step", json={"decision": decision, "reason": reason})
        resp.raise_for_status()
        result = resp.json()

        obs    = result["observation"]
        reward = result["reward"]
        total_reward += reward

        # STEP log (required format)
        print(f"[STEP] {step} | {obs.get('invoice_id', 'N/A')} | {decision} | {reward:.1f}")

    resp = requests.post(f"{ENV_URL}/grader")
    resp.raise_for_status()
    score = resp.json()["score"]

    # END log (required format)
    print(f"[END] {task_name} | {score:.2f}")

    return score


def main():
    resp = requests.get(f"{ENV_URL}/tasks")
    resp.raise_for_status()
    tasks = [t["name"] for t in resp.json()["tasks"]]

    scores = {}
    for task in tasks:
        scores[task] = run_task(task)

    print("\n=== FINAL SCORES ===")
    for task, score in scores.items():
        print(f"{task}: {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"Average: {avg:.2f}")

    return scores


if __name__ == "__main__":
    main()
