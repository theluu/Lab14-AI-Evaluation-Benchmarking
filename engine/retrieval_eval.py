"""
Đánh giá tầng Retrieval: Hit Rate & MRR.

- Hit Rate@k: có ít nhất 1 tài liệu đúng nằm trong top-k không (1.0/0.0).
- MRR:        nghịch đảo thứ hạng của tài liệu đúng đầu tiên (vị trí 1 -> 1.0, vị trí 2 -> 0.5...).

Lưu ý: case 'out-of-context' có expected_ids rỗng (không tài liệu nào trả lời được).
Khi đó metric retrieval không áp dụng -> trả None và bị loại khỏi trung bình.
"""
from typing import List, Dict, Optional


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def score_case(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> Dict[str, Optional[float]]:
        """Chấm 1 case. expected_ids rỗng => metric không áp dụng (None)."""
        if not expected_ids:
            return {"hit_rate": None, "mrr": None}
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
        }

    def aggregate(self, per_case: List[Dict[str, Optional[float]]]) -> Dict[str, float]:
        hits = [c["hit_rate"] for c in per_case if c["hit_rate"] is not None]
        mrrs = [c["mrr"] for c in per_case if c["mrr"] is not None]
        return {
            "avg_hit_rate": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "avg_mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "evaluated_cases": len(hits),
        }
