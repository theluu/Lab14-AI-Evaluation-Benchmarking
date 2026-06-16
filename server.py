"""
Web UI backend (Flask) cho AI Evaluation Factory.

Bọc engine bất đồng bộ bằng MỘT event loop chạy nền (background thread) để có thể
tái sử dụng AsyncOpenAI client xuyên suốt nhiều request (tránh lỗi 'event loop closed').

Chạy:  python server.py   ->   http://127.0.0.1:5000
"""
import asyncio
import json
import os
import threading
import time

from flask import Flask, request, jsonify, send_from_directory

from engine import config
from engine.cost import CostTracker
from engine.retriever import Retriever
from engine.faithfulness import FaithfulnessEvaluator
from engine.llm_judge import MultiJudge
from agent.main_agent import MainAgent

# ---------------- event loop nền ----------------
_loop = asyncio.new_event_loop()
threading.Thread(target=lambda: (asyncio.set_event_loop(_loop), _loop.run_forever()),
                 daemon=True).start()


def run_async(coro):
    """Chạy coroutine trên event loop nền, chờ kết quả (đồng bộ với Flask)."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ---------------- thành phần dùng chung ----------------
_lock = threading.Lock()
_state = {"retriever": None, "agents": {}}
_faith = FaithfulnessEvaluator()
_judge = MultiJudge()


def get_retriever():
    with _lock:
        if _state["retriever"] is None:
            r = Retriever()
            run_async(r.build_index())
            _state["retriever"] = r
        return _state["retriever"]


def get_agent(version):
    get_retriever()
    with _lock:
        if version not in _state["agents"]:
            _state["agents"][version] = MainAgent(version=version, retriever=_state["retriever"])
        return _state["agents"][version]


app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.get("/api/status")
def api_status():
    return jsonify({
        "mode": "MOCK" if config.MOCK_MODE else "LIVE",
        "agent_model": config.AGENT_MODEL,
        "judge_a": config.JUDGE_A_MODEL,
        "judge_b": config.JUDGE_B_MODEL,
    })


# Câu hỏi có sẵn, phủ hết 14 tài liệu trong corpus (tất cả trả lời được từ RAG)
ASK_QUESTIONS = [
    {"kb": "KB-001", "q": "Làm thế nào để đổi mật khẩu công ty?"},
    {"kb": "KB-001", "q": "Mật khẩu mới cần đáp ứng những yêu cầu gì?"},
    {"kb": "KB-002", "q": "Tài khoản bị khóa bao lâu thì tự mở?"},
    {"kb": "KB-002", "q": "Tôi nhập sai mật khẩu mấy lần thì bị khóa?"},
    {"kb": "KB-003", "q": "Làm sao để kết nối VPN của công ty?"},
    {"kb": "KB-003", "q": "VPN báo lỗi 'Authentication failed' thì phải làm gì?"},
    {"kb": "KB-004", "q": "Cách thiết lập email công ty trên điện thoại?"},
    {"kb": "KB-005", "q": "Làm thế nào để bật xác thực hai lớp (MFA)?"},
    {"kb": "KB-005", "q": "Mất điện thoại đăng ký MFA thì xử lý thế nào?"},
    {"kb": "KB-006", "q": "Kết nối Wi-Fi văn phòng như thế nào?"},
    {"kb": "KB-006", "q": "Mạng Wi-Fi dành cho khách tên gì và lấy mật khẩu ở đâu?"},
    {"kb": "KB-007", "q": "Tôi muốn cài một phần mềm mới thì làm sao?"},
    {"kb": "KB-008", "q": "Lệnh in bị treo thì xử lý thế nào?"},
    {"kb": "KB-008", "q": "Máy in báo hết mực thì báo cho ai?"},
    {"kb": "KB-009", "q": "Dữ liệu lưu ở Desktop có được sao lưu không?"},
    {"kb": "KB-009", "q": "Làm sao khôi phục file đã lỡ xóa?"},
    {"kb": "KB-010", "q": "Cách truy cập máy tính công ty từ xa?"},
    {"kb": "KB-011", "q": "Làm sao nhận biết một email lừa đảo (phishing)?"},
    {"kb": "KB-012", "q": "Nhân viên mới được cấp laptop khi nào?"},
    {"kb": "KB-013", "q": "Ticket ưu tiên P1 được phản hồi trong bao lâu?"},
    {"kb": "KB-013", "q": "Giờ hỗ trợ của Helpdesk là khi nào?"},
    {"kb": "KB-014", "q": "Tôi có được dùng USB cá nhân ở công ty không?"},
]

# Preset cho Multi-Judge: bấm là tự điền sẵn Q / đáp án chuẩn / câu trả lời cần chấm
JUDGE_PRESETS = [
    {"label": "✅ Câu trả lời ĐÚNG",
     "q": "Tài khoản bị khóa bao lâu thì tự mở?",
     "gt": "Tài khoản tự mở sau 30 phút; nếu cần mở ngay thì gọi Helpdesk nội bộ 1234.",
     "a": "Tài khoản sẽ tự động mở khóa sau 30 phút. Nếu bạn cần mở ngay, hãy gọi IT Helpdesk theo số nội bộ 1234 và xác minh danh tính."},
    {"label": "❌ Câu trả lời SAI",
     "q": "Làm thế nào để đổi mật khẩu công ty?",
     "gt": "Vào portal.company.local chọn 'Đổi mật khẩu'; tối thiểu 12 ký tự, hết hạn sau 90 ngày.",
     "a": "Bạn cứ gọi điện trực tiếp cho sếp nhờ đổi giúp là xong."},
    {"label": "➗ Đúng một phần",
     "q": "Mật khẩu mới cần đáp ứng yêu cầu gì?",
     "gt": "Tối thiểu 12 ký tự, gồm chữ hoa, thường, số, ký tự đặc biệt; không dùng lại 5 mật khẩu gần nhất; hết hạn 90 ngày.",
     "a": "Mật khẩu cần dài ít nhất 12 ký tự."},
]


@app.get("/api/samples")
def api_samples():
    """Câu hỏi/preset có sẵn để bấm nhanh (khỏi gõ)."""
    traps = []
    path = "data/golden_set.jsonl"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c["metadata"].get("type") == "hallucination_trap" and len(traps) < 6:
                    traps.append({"q": c["question"], "gt": c["expected_answer"]})
    if not traps:
        traps = [{"q": "VPN GlobalConnect cho đăng nhập tối đa bao nhiêu thiết bị cùng lúc?",
                  "gt": "Tài liệu không nêu giới hạn; agent không được bịa con số."}]
    return jsonify({"ask": ASK_QUESTIONS, "traps": traps, "judge_presets": JUDGE_PRESETS})


@app.post("/api/ask")
def api_ask():
    data = request.get_json(force=True)
    q = (data.get("question") or "").strip()
    version = data.get("version", "V2")
    if not q:
        return jsonify({"error": "Thiếu câu hỏi"}), 400
    agent = get_agent(version)
    tracker = CostTracker()
    t0 = time.perf_counter()
    resp = run_async(agent.query(q, tracker=tracker))
    latency = round(time.perf_counter() - t0, 2)
    return jsonify({
        "version": version,
        "answer": resp["answer"],
        "retrieved_ids": resp["retrieved_ids"],
        "contexts": resp["contexts"],
        "latency": latency,
        "cost": tracker.summary(1)["total_cost_usd"],
    })


@app.post("/api/compare")
def api_compare():
    """Chạy CẢ V1 và V2 trên cùng câu hỏi + chấm faithfulness -> demo hallucination."""
    data = request.get_json(force=True)
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"error": "Thiếu câu hỏi"}), 400
    tracker = CostTracker()
    out = {}
    for v in ("V1", "V2"):
        agent = get_agent(v)
        resp = run_async(agent.query(q, tracker=tracker))
        f = run_async(_faith.score(q, resp["answer"], resp["contexts"], tracker=tracker))
        out[v] = {
            "answer": resp["answer"],
            "faithfulness": f["faithfulness"],
            "relevancy": f["relevancy"],
            "retrieved_ids": resp["retrieved_ids"],
        }
    out["cost"] = tracker.summary(1)["total_cost_usd"]
    return jsonify(out)


@app.post("/api/judge")
def api_judge():
    data = request.get_json(force=True)
    q = (data.get("question") or "").strip()
    a = (data.get("answer") or "").strip()
    gt = (data.get("ground_truth") or "").strip()
    if not (q and a):
        return jsonify({"error": "Cần cả câu hỏi và câu trả lời"}), 400
    tracker = CostTracker()
    res = run_async(_judge.evaluate_multi_judge(q, a, gt or "(không có đáp án chuẩn)", tracker=tracker))
    res["cost"] = tracker.summary(1)["total_cost_usd"]
    return jsonify(res)


@app.get("/api/summary")
def api_summary():
    path = "reports/summary.json"
    if not os.path.exists(path):
        return jsonify({"error": "Chưa có reports/summary.json — hãy chạy: python main.py"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    # macOS chiếm cổng 5000 cho AirPlay Receiver -> dùng 5050. Đổi bằng biến PORT nếu cần.
    port = int(os.getenv("PORT", "5050"))
    print(config.mode_banner())
    print(f"🌐 Console mở tại:  http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
