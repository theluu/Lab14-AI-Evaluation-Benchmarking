"""
Theo dõi Chi phí & Token usage cho toàn pipeline.

CostTracker tích lũy số token in/out theo từng model và quy ra USD.
Hỗ trợ tính 'Giá tiền cho mỗi lần Eval' theo yêu cầu Expert.
"""
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict

from engine.config import PRICING


@dataclass
class CostTracker:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: Dict[str, dict] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        in_price, out_price = PRICING.get(model, (0.0, 0.0))
        cost = prompt_tokens * in_price + completion_tokens * out_price
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.cost_usd += cost
            m = self.by_model.setdefault(
                model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            )
            m["calls"] += 1
            m["prompt_tokens"] += prompt_tokens
            m["completion_tokens"] += completion_tokens
            m["cost_usd"] += cost
        return cost

    def summary(self, num_cases: int = 0) -> dict:
        total_tokens = self.prompt_tokens + self.completion_tokens
        return {
            "total_llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": round(self.cost_usd, 6),
            "cost_per_eval_usd": round(self.cost_usd / num_cases, 6) if num_cases else 0.0,
            "by_model": {
                k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in self.by_model.items()
            },
        }
