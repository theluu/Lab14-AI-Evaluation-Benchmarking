"""
Lớp bọc (wrapper) cho các lời gọi OpenAI dùng chung toàn hệ thống.

- chat_json(): gọi chat completion, ép trả JSON, tự ghi nhận token/cost.
- embed(): gọi embedding, có fallback deterministic khi MOCK_MODE.
- Khi MOCK_MODE: trả kết quả giả lập có kiểm soát để pipeline vẫn chạy.
"""
import hashlib
import json
import math
from typing import List, Optional

from engine import config
from engine.cost import CostTracker


def _hash_vector(text: str, dim: int = 256) -> List[float]:
    """Sinh vector giả lập deterministic từ text (dùng cho MOCK_MODE).

    Băm từng token vào các chiều => văn bản giống nhau cho vector giống nhau,
    đủ để cosine-similarity phân biệt được câu liên quan/không liên quan.
    """
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def embed(texts: List[str], tracker: Optional[CostTracker] = None) -> List[List[float]]:
    """Embed danh sách text. MOCK_MODE => vector băm deterministic."""
    if config.MOCK_MODE:
        return [_hash_vector(t) for t in texts]

    client = config.get_client()
    resp = await client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    if tracker is not None:
        tracker.record(config.EMBED_MODEL, resp.usage.prompt_tokens, 0)
    return [d.embedding for d in resp.data]


def _mock_json_response(system: str, user: str, model: str = "") -> dict:
    """Sinh phản hồi JSON giả lập deterministic dựa trên nội dung prompt + model.

    Đưa model vào seed để 2 judge khác model cho điểm hơi khác nhau -> mô phỏng
    được agreement < 1.0 và nhánh xử lý xung đột."""
    seed = int(hashlib.md5((system + user + model).encode()).hexdigest(), 16)
    # điểm 3..5 ổn định theo nội dung
    score = 3 + (seed % 3)
    return {
        "_mock": True,
        "score": score,
        "accuracy": score,
        "faithfulness": round(0.6 + (seed % 40) / 100, 2),
        "relevancy": round(0.6 + ((seed >> 3) % 40) / 100, 2),
        "answer": "[MOCK] Câu trả lời giả lập dựa trên context được cung cấp.",
        "reasoning": "[MOCK] Lý giải giả lập (deterministic).",
        "questions": [],
    }


async def chat_json(
    model: str,
    system: str,
    user: str,
    tracker: Optional[CostTracker] = None,
    temperature: float = 0.0,
) -> dict:
    """Gọi chat completion và parse JSON. Tự ghi nhận cost.

    MOCK_MODE => trả JSON giả lập deterministic (không gọi mạng).
    """
    if config.MOCK_MODE:
        return _mock_json_response(system, user, model)

    client = config.get_client()
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tracker is not None:
        tracker.record(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": content}


async def chat_text(
    model: str,
    system: str,
    user: str,
    tracker: Optional[CostTracker] = None,
    temperature: float = 0.2,
) -> str:
    """Gọi chat completion trả text thường (dùng cho Agent sinh câu trả lời)."""
    if config.MOCK_MODE:
        return ("[MOCK] Dựa trên tài liệu hỗ trợ, đây là câu trả lời giả lập "
                "cho câu hỏi của bạn.")

    client = config.get_client()
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tracker is not None:
        tracker.record(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content
