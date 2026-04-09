"""
Inference Script — Invoice Processing Agent
===========================================
MANDATORY variables (set in environment):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face API key.
"""

import os
import json
import requests
import time
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://127.0.0.1:7860")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Structured log helpers ─────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done}{error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)

# ── LLM Logic with Retries ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert invoice processing agent. Decision must be exactly one of: approve, reject, flag.
Respond ONLY with JSON: {"decision": "approve", "reason": "reason"}"""

def call_llm(invoice_obs: dict, retries=3) -> dict:
    prompt = f"Process this invoice: {json.dumps(invoice_obs)}"
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                timeout=30.0  # Prevent hanging forever
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            if attempt == retries - 1:
                print(f"Log: LLM failed after {retries} attempts: {e}")
                return {"decision": "flag", "reason": f"LLM Error: {str(e)}"}
            time.sleep(2) # Wait before retry
    return {"decision": "flag", "reason": "Max retries exceeded"}

# ── Task Runner ─────────────────────────────────────────────────────────────────
def run_task(task_name: str) -> float:
    rewards = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_name, env="invoice-processing-agent", model=MODEL_NAME)

    try:
        # Initial reset
        try:
            resp = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": 42}, timeout=10)
            resp.raise_for_status()
            obs = resp.json()
        except Exception as e:
            print(f"Log: Failed to reset environment: {e}")
            log_end(False, 0, 0.01, [])
            return 0.01

        for step in range(1, 20):
            if obs.get("done", False):
                break

            # CALL LLM (Wrapped in retries)
            action = call_llm(obs)
            decision = action.get("decision", "flag").lower().strip()
            reason = action.get("reason", "no reason")
            
            # SUBMIT STEP
            try:
                resp = requests.post(f"{ENV_URL}/step", json={"decision": decision, "reason": reason}, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                obs = result["observation"]
                reward = result["reward"]
                done = result["done"]
            except Exception as e:
                log_step(step, decision, 0.01, False, error=str(e))
                break

            rewards.append(reward)
            steps_taken = step
            log_step(step, decision, reward, done)

            if done:
                break

        # GET FINAL SCORE
        try:
            resp = requests.post(f"{ENV_URL}/grader", timeout=10)
            resp.raise_for_status()
            score = max(0.01, min(0.99, float(resp.json()["score"])))
        except:
            score = 0.01
            
        success = score >= 0.5

    finally:
        log_end(success, steps_taken, score, rewards)

    return score

def main():
    try:
        resp = requests.get(f"{ENV_URL}/tasks", timeout=10)
        resp.raise_for_status()
        tasks = [t["name"] for t in resp.json()["tasks"]]
    except:
        tasks = ["easy_triage", "medium_triage", "hard_triage"]

    scores = {}
    for task in tasks:
        scores[task] = run_task(task)

    print("\n=== FINAL SCORES ===", flush=True)
    for task, score in scores.items():
        print(f"{task}: {score:.4f}", flush=True)

if __name__ == "__main__":
    main()