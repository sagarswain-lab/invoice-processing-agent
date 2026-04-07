from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import InvoiceAction, InvoiceObservation, InvoiceState
from environment import InvoiceProcessingEnvironment, TASKS

app = FastAPI(title="Invoice Processing Agent — OpenEnv", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ✅ Fixed path — index.html is directly inside server/
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "index.html")
env = InvoiceProcessingEnvironment()

class ResetRequest(BaseModel):
    task_name: str = "easy_triage"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    decision: str
    reason: str = ""

@app.get("/", response_class=HTMLResponse)
def root():
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, media_type="text/html; charset=utf-8")

@app.get("/favicon.ico")
def favicon():
    return None # Silent response to prevent 404 logs

@app.post("/reset", response_model=InvoiceObservation)
def reset(req: Optional[ResetRequest] = None):
    """Start a new episode. Body is optional — defaults to easy_triage."""
    try:
        task_name = req.task_name if req else "easy_triage"
        seed = req.seed if req else None
        return env.reset(task_name=task_name, seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(req: StepRequest):
    if req.decision.lower() not in {"approve", "reject", "flag"}:
        raise HTTPException(status_code=400, detail="decision must be approve, reject, or flag")
    action = InvoiceAction(decision=req.decision, reason=req.reason)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@app.get("/state", response_model=InvoiceState)
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [{"name": n, "description": m["description"], "difficulty": m["difficulty"], "num_invoices": len(m["invoices"])} for n, m in TASKS.items()],
        "action_schema": {"decision": {"type": "string", "allowed_values": ["approve", "reject", "flag"]}, "reason": {"type": "string"}}
    }

@app.post("/grader")
def grader():
    score = env.grade()
    s = env.state()
    return {"score": score, "correct_decisions": s.correct_decisions, "total_invoices": s.total_invoices, "task_name": s.task_name}

@app.post("/baseline")
def baseline():
    """Run rule-based agent on all 3 tasks. Returns reproducible scores."""
    results = {}
    for task_name in TASKS:
        obs = env.reset(task_name=task_name, seed=42)
        total_reward = 0.0
        while not obs.done:
            if "unknown_vendor" in obs.flags or obs.amount > 20000:
                decision = "reject"
            elif len(obs.flags) > 0:
                decision = "flag"
            else:
                decision = "approve"
            action = InvoiceAction(decision=decision, reason="baseline rule")
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        results[task_name] = {"score": env.grade(), "total_reward": round(total_reward, 2)}
    return {"baseline_scores": results}

@app.get("/health")
def health():
    return {"status": "ok", "environment": "invoice-processing-agent"}


def main():
    """Entry point for multi-mode deployment (uv run / script execution)."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()