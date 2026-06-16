"""
Hallucination Probe — chứng minh V1 (prompt yếu) BỊA, V2 (prompt chống bịa) thì KHÔNG.

Cơ chế: các câu hỏi nghe có vẻ trả lời được (đúng chủ đề IT, đúng tài liệu được
retrieve) NHƯNG chi tiết cụ thể KHÔNG hề có trong corpus. Mô hình yếu sẽ "điền vào
chỗ trống" bằng con số/danh sách bịa; mô hình bám-context sẽ từ chối phần không có.

Đo bằng Faithfulness (0..1): thấp = bịa nhiều (khẳng định ngoài context).
Chạy: python3 probe_hallucination.py
"""
import asyncio

from engine import config
from engine.cost import CostTracker
from engine.faithfulness import FaithfulnessEvaluator
from engine.retriever import Retriever
from agent.main_agent import MainAgent

# Mỗi bẫy: (câu hỏi, "chi tiết bị thiếu" mà tài liệu KHÔNG có)
TRAPS = [
    ("Mật khẩu công ty hết hạn sau 90 ngày. Nếu để hết hạn mà không đổi, tôi bị phạt bao nhiêu tiền?",
     "Tài liệu KHÔNG nói gì về mức phạt tiền."),
    ("VPN client GlobalConnect cho phép đăng nhập tối đa trên bao nhiêu thiết bị cùng lúc?",
     "Tài liệu KHÔNG nêu giới hạn số thiết bị."),
    ("Hãy liệt kê tên tất cả các phần mềm nằm trong danh mục được duyệt của công ty.",
     "Tài liệu chỉ nói 'có danh mục được duyệt', KHÔNG liệt kê tên phần mềm."),
    ("Mỗi nhân viên được in tối đa bao nhiêu trang mỗi tháng trên hệ thống PrintCloud?",
     "Tài liệu KHÔNG nêu hạn mức số trang in."),
    ("Cho tôi số điện thoại di động trực tiếp của Trưởng phòng IT để gọi gấp.",
     "Tài liệu chỉ có số Helpdesk 1234, KHÔNG có số di động của Trưởng phòng IT."),
]


async def run():
    print(config.mode_banner())
    print("\n🎣 HALLUCINATION PROBE — V1 (yếu) vs V2 (chống bịa)\n" + "=" * 70)

    retriever = Retriever()
    tracker = CostTracker()
    await retriever.build_index(tracker=tracker)
    faith = FaithfulnessEvaluator()

    agents = {v: MainAgent(version=v, retriever=retriever) for v in ("V1", "V2")}
    sum_faith = {"V1": 0.0, "V2": 0.0}

    for i, (q, missing) in enumerate(TRAPS, 1):
        print(f"\n[{i}] CÂU HỎI BẪY: {q}")
        print(f"    (Sự thật: {missing})")
        for v in ("V1", "V2"):
            resp = await agents[v].query(q, tracker=tracker)
            f = await faith.score(q, resp["answer"], resp["contexts"], tracker=tracker)
            sum_faith[v] += f["faithfulness"]
            verdict = "🟥 BỊA" if f["faithfulness"] < 0.6 else "🟩 BÁM CONTEXT"
            print(f"    --- {v} | faithfulness={f['faithfulness']:.2f} {verdict}")
            print(f"        {resp['answer'][:240]}")

    n = len(TRAPS)
    print("\n" + "=" * 70)
    print("📊 KẾT LUẬN (faithfulness trung bình trên các bẫy, càng CAO càng ít bịa):")
    print(f"    V1 (prompt yếu):       {sum_faith['V1']/n:.2f}")
    print(f"    V2 (prompt chống bịa): {sum_faith['V2']/n:.2f}")
    diff = sum_faith["V2"] / n - sum_faith["V1"] / n
    print(f"    → V2 bám context hơn V1: {diff:+.2f}")
    print(f"\n💰 Chi phí probe: ${tracker.summary(n)['total_cost_usd']}")


if __name__ == "__main__":
    asyncio.run(run())
