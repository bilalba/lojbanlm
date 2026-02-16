"""Track API costs per model with budget enforcement."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import COST_LOG_PATH, PRICING


class CostTracker:
    """Track and report API costs. Backed by append-only JSONL."""

    def __init__(self, log_path: Path = COST_LOG_PATH, budget_usd: float = 0.0):
        self._log_path = log_path
        self._budget = budget_usd  # 0 = unlimited
        self._total_cost = 0.0
        self._by_model: dict[str, float] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._load_existing()

    def _load_existing(self):
        if not self._log_path.exists():
            return
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cost = entry.get("cost_usd", 0.0)
                model = entry.get("model", "unknown")
                self._total_cost += cost
                self._by_model[model] = self._by_model.get(model, 0.0) + cost
                self._total_input_tokens += entry.get("input_tokens", 0)
                self._total_output_tokens += entry.get("output_tokens", 0)

    def record(self, chunk_id: str, model: str, input_tokens: int, output_tokens: int):
        """Record a translation's cost. Append to cost_log.jsonl."""
        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        self._total_cost += cost
        self._by_model[model] = self._by_model.get(model, 0.0) + cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        entry = {
            "chunk_id": chunk_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def check_budget(self) -> bool:
        """Return True if within budget (or no budget set)."""
        if self._budget <= 0:
            return True
        return self._total_cost < self._budget

    def summary(self) -> dict:
        """Return cost breakdown."""
        return {
            "total_cost_usd": round(self._total_cost, 4),
            "budget_usd": self._budget,
            "by_model": {k: round(v, 4) for k, v in self._by_model.items()},
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    def print_summary(self):
        """Print human-readable cost summary."""
        s = self.summary()
        print(f"\n=== Cost Summary ===")
        print(f"Total: ${s['total_cost_usd']:.4f}")
        if s["budget_usd"] > 0:
            print(f"Budget: ${s['budget_usd']:.2f} ({s['total_cost_usd']/s['budget_usd']*100:.1f}% used)")
        print(f"Tokens: {s['total_input_tokens']:,} in / {s['total_output_tokens']:,} out")
        for model, cost in s["by_model"].items():
            print(f"  {model}: ${cost:.4f}")
