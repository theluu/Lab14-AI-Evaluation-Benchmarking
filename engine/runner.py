"""
BenchmarkRunner — điều phối chạy đánh giá cho toàn bộ dataset.

Mỗi case đi qua: Agent -> Retrieval eval -> Faithfulness -> Multi-Judge.
Chạy SONG SONG bằng asyncio + Semaphore (giới hạn concurrency để không bị Rate Limit).
Đo latency, token và cost cho từng lần eval.
"""
import asyncio
import time
from typing import List, Dict

from engine import config
from engine.cost import CostTracker
from engine.faithfulness import FaithfulnessEvaluator
from engine.llm_judge import MultiJudge
from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    def __init__(self, agent, judge: MultiJudge = None, tracker: CostTracker = None):
        self.agent = agent
        self.retrieval_eval = RetrievalEvaluator()
        self.faithfulness = FaithfulnessEvaluator()
        self.judge = judge or MultiJudge()
        self.tracker = tracker or CostTracker()
        self._sem = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def run_single_test(self, case: Dict) -> Dict:
        async with self._sem:
            start = time.perf_counter()
            resp = await self.agent.query(case["question"], tracker=self.tracker)
            latency = time.perf_counter() - start

            retrieval = self.retrieval_eval.score_case(
                case.get("expected_retrieval_ids", []),
                resp["retrieved_ids"],
                top_k=self.agent.cfg["top_k"],
            )
            faith = await self.faithfulness.score(
                case["question"], resp["answer"], resp["contexts"], tracker=self.tracker
            )
            judge = await self.judge.evaluate_multi_judge(
                case["question"], resp["answer"], case["expected_answer"], tracker=self.tracker
            )

        return {
            "id": case.get("id"),
            "type": case["metadata"].get("type"),
            "difficulty": case["metadata"].get("difficulty"),
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "agent_answer": resp["answer"],
            "expected_retrieval_ids": case.get("expected_retrieval_ids", []),
            "retrieved_ids": resp["retrieved_ids"],
            "latency_s": round(latency, 3),
            "retrieval": retrieval,
            "faithfulness": faith["faithfulness"],
            "relevancy": faith["relevancy"],
            "judge": judge,
            "status": "pass" if judge["final_score"] >= 3 else "fail",
        }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        tasks = [self.run_single_test(case) for case in dataset]
        return await asyncio.gather(*tasks)
