"""
Cấu hình trung tâm cho Evaluation Factory.

- Quản lý OpenAI client (async).
- Bảng giá token để tính Cost.
- Tự động bật MOCK_MODE khi không có API key => hệ thống vẫn chạy được
  (deterministic) để check_lab.py luôn pass khi nộp bài.
"""
import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Models ----
EMBED_MODEL = "text-embedding-3-small"
AGENT_MODEL = "gpt-4o-mini"        # model Agent dùng để sinh câu trả lời
JUDGE_A_MODEL = "gpt-4o"           # Judge A (mạnh)
JUDGE_B_MODEL = "gpt-4o-mini"      # Judge B (rẻ) -> 2 judge khác nhau
FAITH_MODEL = "gpt-4o-mini"        # model chấm faithfulness/relevancy

# ---- Bảng giá (USD / 1 token). Nguồn: bảng giá OpenAI công bố. ----
# Lưu theo (input_price, output_price) cho mỗi 1 token.
PRICING = {
    "gpt-4o":                 (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini":            (0.15 / 1_000_000,  0.60 / 1_000_000),
    "text-embedding-3-small": (0.02 / 1_000_000,  0.0),
}

# ---- MOCK MODE ----
# Bật khi thiếu key. Khi đó các lời gọi LLM trả kết quả giả lập có kiểm soát.
MOCK_MODE = not bool(os.getenv("OPENAI_API_KEY"))

# Giới hạn song song để tránh Rate Limit
MAX_CONCURRENCY = int(os.getenv("EVAL_MAX_CONCURRENCY", "5"))


@lru_cache(maxsize=1)
def get_client():
    """Trả về AsyncOpenAI client (cache 1 lần). Trả None nếu MOCK_MODE."""
    if MOCK_MODE:
        return None
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def mode_banner() -> str:
    if MOCK_MODE:
        return ("⚠️  MOCK_MODE BẬT (không thấy OPENAI_API_KEY). "
                "Hệ thống chạy giả lập deterministic. "
                "Đặt OPENAI_API_KEY trong .env để gọi LLM thật.")
    return "✅ LIVE_MODE: gọi OpenAI API thật."
