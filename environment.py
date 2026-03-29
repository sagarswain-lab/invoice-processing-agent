import uuid
import random
from typing import Optional, Tuple
from models import InvoiceAction, InvoiceObservation, InvoiceState

# ─────────────────────────────────────────────
# Synthetic invoice data for each task
# ─────────────────────────────────────────────

EASY_INVOICES = [
    {
        "invoice_id": "INV-1001",
        "vendor": "Office Supplies Co.",
        "amount": 450.00,
        "currency": "USD",
        "date": "2024-03-01",
        "line_items": [{"desc": "Pens x50", "qty": 50, "unit_price": 5.0, "total": 250.0},
                       {"desc": "Notebooks x20", "qty": 20, "unit_price": 10.0, "total": 200.0}],
        "correct_decision": "approve",
        "flags": []
    },
    {
        "invoice_id": "INV-1002",
        "vendor": "Unknown Vendor XYZ",
        "amount": 99999.99,
        "currency": "USD",
        "date": "2024-03-02",
        "line_items": [{"desc": "Misc services", "qty": 1, "unit_price": 99999.99, "total": 99999.99}],
        "correct_decision": "reject",
        "flags": ["amount_anomaly", "unknown_vendor"]
    },
    {
        "invoice_id": "INV-1003",
        "vendor": "Tech Hardware Ltd.",
        "amount": 1200.00,
        "currency": "USD",
        "date": "2024-03-03",
        "line_items": [{"desc": "Laptop Stand", "qty": 4, "unit_price": 300.0, "total": 1200.0}],
        "correct_decision": "approve",
        "flags": []
    },
]

MEDIUM_INVOICES = [
    {
        "invoice_id": "INV-2001",
        "vendor": "CloudSoft Inc.",
        "amount": 5000.00,
        "currency": "USD",
        "date": "2024-03-10",
        "line_items": [{"desc": "Annual SaaS license", "qty": 1, "unit_price": 5000.0, "total": 5000.0}],
        "correct_decision": "approve",
        "flags": []
    },
    {
        "invoice_id": "INV-2002",
        "vendor": "CloudSoft Inc.",
        "amount": 5000.00,
        "currency": "USD",
        "date": "2024-03-10",
        "line_items": [{"desc": "Annual SaaS license", "qty": 1, "unit_price": 5000.0, "total": 5000.0}],
        "correct_decision": "reject",
        "flags": ["duplicate"]  # Same as INV-2001
    },
    {
        "invoice_id": "INV-2003",
        "vendor": "Marketing Agency Pro",
        "amount": 8500.00,
        "currency": "USD",
        "date": "2024-03-11",
        "line_items": [{"desc": "Campaign Q1", "qty": 1, "unit_price": 8000.0, "total": 8000.0},
                       {"desc": "Design assets", "qty": 1, "unit_price": 500.0, "total": 500.0}],
        "correct_decision": "flag",
        "flags": ["amount_anomaly"]  # Over budget threshold
    },
    {
        "invoice_id": "INV-2004",
        "vendor": "Catering Services",
        "amount": 320.00,
        "currency": "USD",
        "date": "2024-03-12",
        "line_items": [{"desc": "Team lunch", "qty": 1, "unit_price": 320.0, "total": 320.0}],
        "correct_decision": "approve",
        "flags": []
    },
]

HARD_INVOICES = [
    {
        "invoice_id": "INV-3001",
        "vendor": "Acme Consulting",
        "amount": 12000.00,
        "currency": "USD",
        "date": "2024-03-20",
        "line_items": [{"desc": "Consulting hours", "qty": 40, "unit_price": 250.0, "total": 10000.0},
                       {"desc": "Expenses", "qty": 1, "unit_price": 2000.0, "total": 2000.0}],
        "correct_decision": "flag",
        "flags": ["missing_field", "amount_anomaly"]  # Line items don't add up correctly
    },
    {
        "invoice_id": "INV-3002",
        "vendor": "Legit Supplies Ltd.",
        "amount": 750.00,
        "currency": "EUR",   # Currency mismatch
        "date": "2024-03-21",
        "line_items": [{"desc": "Office chairs", "qty": 3, "unit_price": 250.0, "total": 750.0}],
        "correct_decision": "flag",
        "flags": ["currency_mismatch"]
    },
    {
        "invoice_id": "INV-3003",
        "vendor": "",  # Missing vendor
        "amount": 3400.00,
        "currency": "USD",
        "date": "2024-03-22",
        "line_items": [{"desc": "Server parts", "qty": 2, "unit_price": 1700.0, "total": 3400.0}],
        "correct_decision": "reject",
        "flags": ["missing_field"]
    },
    {
        "invoice_id": "INV-3004",
        "vendor": "Trusted Vendor Co.",
        "amount": 980.00,
        "currency": "USD",
        "date": "2024-03-23",
        "line_items": [{"desc": "Printer cartridges", "qty": 10, "unit_price": 98.0, "total": 980.0}],
        "correct_decision": "approve",
        "flags": []
    },
    {
        "invoice_id": "INV-3005",
        "vendor": "Shady Deals LLC",
        "amount": 50000.00,
        "currency": "USD",
        "date": "2024-03-24",
        "line_items": [{"desc": "Vague services", "qty": 1, "unit_price": 50000.0, "total": 50000.0}],
        "correct_decision": "reject",
        "flags": ["unknown_vendor", "amount_anomaly"]
    },
]

TASKS = {
    "easy_triage": {
        "description": "Approve or reject straightforward invoices. Clear signals provided.",
        "invoices": EASY_INVOICES,
        "difficulty": "easy"
    },
    "medium_triage": {
        "description": "Handle duplicates, over-budget invoices, and multi-vendor scenarios.",
        "invoices": MEDIUM_INVOICES,
        "difficulty": "medium"
    },
    "hard_triage": {
        "description": "Detect subtle anomalies: missing fields, currency mismatches, line item errors.",
        "invoices": HARD_INVOICES,
        "difficulty": "hard"
    }
}


class InvoiceProcessingEnvironment:
    """
    OpenEnv environment simulating an invoice processing pipeline.

    An AI agent reads invoices one by one and decides to:
    - approve: process for payment
    - reject: send back to vendor
    - flag: escalate for human review

    Three tasks of increasing difficulty.
    """

    def __init__(self):
        self._task_name: Optional[str] = None
        self._invoices: list = []
        self._current_index: int = 0
        self._episode_id: str = ""
        self._correct_decisions: int = 0
        self._step: int = 0

    def reset(self, task_name: str = "easy_triage", seed: Optional[int] = None) -> InvoiceObservation:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")

        if seed is not None:
            random.seed(seed)

        self._task_name = task_name
        self._invoices = TASKS[task_name]["invoices"]
        self._current_index = 0
        self._episode_id = str(uuid.uuid4())[:8]
        self._correct_decisions = 0
        self._step = 0

        return self._make_observation(reward=0.0, done=False, message="Episode started. Process each invoice.")

    def step(self, action: InvoiceAction) -> Tuple[InvoiceObservation, float, bool, dict]:
        invoice = self._invoices[self._current_index]
        correct = invoice["correct_decision"]
        decision = action.decision.lower().strip()

        # Score the decision
        if decision == correct:
            reward = 1.0
            self._correct_decisions += 1
            message = f"✓ Correct! '{decision}' was the right call."
        elif (decision == "flag" and correct in ("approve", "reject")) or \
             (correct == "flag" and decision in ("approve", "reject")):
            reward = 0.3  # Partial credit — flag is cautious
            message = f"~ Partial credit. Expected '{correct}', got '{decision}'."
        else:
            reward = 0.0
            message = f"✗ Wrong. Expected '{correct}', got '{decision}'."

        self._step += 1
        self._current_index += 1
        done = self._current_index >= len(self._invoices)

        obs = self._make_observation(reward=reward, done=done, message=message)
        info = {
            "correct_decision": correct,
            "score_so_far": round(self._correct_decisions / self._step, 2)
        }
        return obs, reward, done, info

    def state(self) -> InvoiceState:
        return InvoiceState(
            episode_id=self._episode_id,
            task_name=self._task_name or "",
            step=self._step,
            total_invoices=len(self._invoices),
            correct_decisions=self._correct_decisions,
            score=round(self._correct_decisions / max(self._step, 1), 2)
        )

    def _make_observation(self, reward: float, done: bool, message: str) -> InvoiceObservation:
        if self._current_index >= len(self._invoices):
            # Episode ended — return summary observation
            return InvoiceObservation(
                invoice_id="DONE",
                vendor="",
                amount=0.0,
                currency="",
                date="",
                line_items=[],
                flags=[],
                task_name=self._task_name or "",
                step=self._step,
                total_steps=len(self._invoices),
                done=True,
                reward=reward,
                message=message
            )

        inv = self._invoices[self._current_index]
        return InvoiceObservation(
            invoice_id=inv["invoice_id"],
            vendor=inv["vendor"],
            amount=inv["amount"],
            currency=inv["currency"],
            date=inv["date"],
            line_items=inv["line_items"],
            flags=inv["flags"],
            task_name=self._task_name or "",
            step=self._step,
            total_steps=len(self._invoices),
            done=done,
            reward=reward,
            message=message
        )

    def grade(self) -> float:
        """Return final score for the episode (0.0 to 1.0)."""
        if self._step == 0:
            return 0.0
        return round(self._correct_decisions / len(self._invoices), 2)
