"""
Đánh giá chất lượng sinh (Generation) theo phong cách RAGAS bằng LLM-judge tự viết:

- Faithfulness: câu trả lời có bám sát (không bịa ngoài) context được cung cấp không? (0..1)
- Answer Relevancy: câu trả lời có đúng trọng tâm câu hỏi không? (0..1)

Dùng custom judge thay vì thư viện RAGAS để chạy ổn định và kiểm soát chi phí.
"""
from typing import Dict, List, Optional

from engine import config, llm_client
from engine.cost import CostTracker

_SYSTEM = (
    "Bạn là giám khảo đánh giá hệ thống RAG. Cho điểm khách quan theo thang 0.0 đến 1.0. "
    "Chỉ trả về JSON."
)


class FaithfulnessEvaluator:
    async def score(
        self, question: str, answer: str, contexts: List[str], tracker: Optional[CostTracker] = None
    ) -> Dict[str, float]:
        context_block = "\n---\n".join(contexts) if contexts else "(không có context)"
        user = (
            f"CÂU HỎI: {question}\n\n"
            f"CONTEXT ĐƯỢC CUNG CẤP:\n{context_block}\n\n"
            f"CÂU TRẢ LỜI CỦA AGENT:\n{answer}\n\n"
            "Đánh giá 2 tiêu chí (0.0-1.0):\n"
            "- faithfulness: mọi khẳng định trong câu trả lời có được context hỗ trợ không "
            "(1.0 = hoàn toàn bám context, 0.0 = bịa đặt). Nếu agent từ chối/nói không biết một cách "
            "hợp lý khi context không có thông tin, faithfulness = 1.0.\n"
            "- relevancy: câu trả lời có đúng trọng tâm câu hỏi không.\n"
            'Trả JSON: {"faithfulness": 0.0, "relevancy": 0.0, "reasoning": "..."}'
        )
        data = await llm_client.chat_json(config.FAITH_MODEL, _SYSTEM, user, tracker=tracker)

        def _clip(x, default=0.0):
            try:
                return max(0.0, min(1.0, float(x)))
            except (TypeError, ValueError):
                return default

        return {
            "faithfulness": _clip(data.get("faithfulness")),
            "relevancy": _clip(data.get("relevancy")),
        }
