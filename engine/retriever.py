"""
Retriever: tầng tìm kiếm tài liệu cho RAG.

- Nạp corpus IT Support, embed toàn bộ chunk (1 lần, có cache).
- Tìm top-k chunk gần nhất theo cosine similarity (numpy in-memory).
- Trả về cả chunk IDs (để tính Hit Rate / MRR) lẫn nội dung (để Agent sinh đáp án).

Không phụ thuộc vector DB ngoài để pipeline gọn nhẹ và chạy được offline.
"""
import json
import os
from typing import List, Dict, Optional

import numpy as np

from engine import llm_client
from engine.cost import CostTracker

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus", "it_support_kb.json")


class Retriever:
    def __init__(self, corpus_path: str = CORPUS_PATH):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus: List[Dict] = json.load(f)
        self.ids = [c["id"] for c in self.corpus]
        self.by_id = {c["id"]: c for c in self.corpus}
        self._matrix: Optional[np.ndarray] = None  # ma trận embedding (n_chunk x dim)

    async def build_index(self, tracker: Optional[CostTracker] = None):
        """Embed toàn bộ corpus một lần và chuẩn hóa vector."""
        texts = [f"{c['title']}. {c['text']}" for c in self.corpus]
        vectors = await llm_client.embed(texts, tracker=tracker)
        mat = np.array(vectors, dtype=np.float32)
        # chuẩn hóa để cosine = tích vô hướng
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms
        return self

    async def retrieve(
        self, query: str, top_k: int = 4, tracker: Optional[CostTracker] = None
    ) -> List[Dict]:
        """Trả về top_k chunk gần nhất: [{id, title, text, score}]."""
        if self._matrix is None:
            await self.build_index(tracker=tracker)
        q_vec = (await llm_client.embed([query], tracker=tracker))[0]
        q = np.array(q_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            c = self.corpus[int(idx)]
            results.append(
                {"id": c["id"], "title": c["title"], "text": c["text"], "score": float(sims[idx])}
            )
        return results
