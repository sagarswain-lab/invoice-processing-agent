from pydantic import BaseModel
from typing import Optional, List


class InvoiceAction(BaseModel):
    """Action the agent takes on an invoice."""
    decision: str  # "approve", "reject", or "flag"
    reason: str    # Short explanation for the decision


class InvoiceObservation(BaseModel):
    """What the agent sees at each step."""
    invoice_id: str
    vendor: str
    amount: float
    currency: str
    date: str
    line_items: List[dict]
    flags: List[str]        # e.g. ["duplicate", "amount_anomaly", "missing_field"]
    task_name: str
    step: int
    total_steps: int
    done: bool = False
    reward: float = 0.0
    message: str = ""


class InvoiceState(BaseModel):
    """Internal state of the environment."""
    episode_id: str
    task_name: str
    step: int
    total_invoices: int
    correct_decisions: int
    score: float
