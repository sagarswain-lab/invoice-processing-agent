from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import InvoiceAction, InvoiceObservation, InvoiceState
from environment import InvoiceProcessingEnvironment, TASKS

app = FastAPI(
    title="Invoice Processing Agent — OpenEnv",
    description="An OpenEnv environment where AI agents process invoices.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.get("/")
def root():
    """Serve the frontend UI."""
    return FileResponse(FRONTEND)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# One shared environment instance (stateful per server)
env = InvoiceProcessingEnvironment()


# ── Request schemas ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "easy_triage"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    decision: str   # "approve", "reject", or "flag"
    reason: str = ""


# ── Core OpenEnv endpoints ─────────────────────────────────────────────────────

@app.post("/reset", response_model=InvoiceObservation)
def reset(req: ResetRequest):
    """Start a new episode for a given task."""
    try:
        obs = env.reset(task_name=req.task_name, seed=req.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Submit a decision on the current invoice."""
    valid = {"approve", "reject", "flag"}
    if req.decision.lower() not in valid:
        raise HTTPException(status_code=400, detail=f"decision must be one of {valid}")

    action = InvoiceAction(decision=req.decision, reason=req.reason)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state", response_model=InvoiceState)
def state():
    """Get current episode state."""
    return env.state()


# ── Required hackathon endpoints ───────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    """List all tasks and the action schema."""
    return {
        "tasks": [
            {
                "name": name,
                "description": meta["description"],
                "difficulty": meta["difficulty"],
                "num_invoices": len(meta["invoices"])
            }
            for name, meta in TASKS.items()
        ],
        "action_schema": {
            "decision": {
                "type": "string",
                "allowed_values": ["approve", "reject", "flag"],
                "description": "The agent's decision on the current invoice"
            },
            "reason": {
                "type": "string",
                "description": "Short explanation for the decision"
            }
        }
    }


@app.post("/grader")
def grader():
    """Return the grader score for the current completed episode (0.0 to 1.0)."""
    score = env.grade()
    state = env.state()
    return {
        "score": score,
        "correct_decisions": state.correct_decisions,
        "total_invoices": state.total_invoices,
        "task_name": state.task_name
    }


@app.post("/baseline")
def baseline():
    """
    Run a simple rule-based baseline agent on all 3 tasks.
    Returns reproducible scores without needing an API key.
    """
    results = {}

    for task_name in TASKS.keys():
        obs = env.reset(task_name=task_name, seed=42)
        total_reward = 0.0

        while not obs.done:
            # Simple rule-based baseline:
            # - Reject if flags contain "unknown_vendor" or "amount_anomaly" > threshold
            # - Flag if any other flags present
            # - Approve otherwise
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

        score = env.grade()
        results[task_name] = {
            "score": score,
            "total_reward": round(total_reward, 2)
        }

    return {"baseline_scores": results}


@app.get("/health")
def health():
    return {"status": "ok", "environment": "invoice-processing-agent"}
