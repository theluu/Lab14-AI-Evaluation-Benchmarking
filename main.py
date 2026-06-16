"""
Điểm vào của Evaluation Factory.

Quy trình:
  1. Nạp Golden Dataset.
  2. Chạy benchmark cho Agent V1 (base) và V2 (optimized) — song song, async.
  3. Tổng hợp metrics: Retrieval (Hit Rate/MRR), Faithfulness, Multi-Judge,
     Agreement/Cohen's Kappa, Latency, Cost.
  4. Kiểm tra Position Bias trên một mẫu.
  5. Regression V1 vs V2 -> Release Gate tự động (APPROVE/BLOCK).
  6. Ghi reports/summary.json + reports/benchmark_results.json.
  7. Sinh analysis/failure_analysis.md từ số liệu thật.
"""
import asyncio
import json
import os
import time

from engine import config, report
from engine.cost import CostTracker
from engine.llm_judge import MultiJudge
from engine.retriever import Retriever
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent

GOLDEN_PATH = "data/golden_set.jsonl"

# Ngưỡng cho Release Gate
GATE_MIN_FAITHFULNESS = 0.60
GATE_MAX_COST_RATIO = 1.50   # V2 không được đắt hơn V1 quá 50%
GATE_SCORE_TOLERANCE = 0.10  # vùng nhiễu: |Δscore| <= 0.1 coi như không đổi (LLM judge có nhiễu)


def load_dataset():
    if not os.path.exists(GOLDEN_PATH):
        print(f"❌ Thiếu {GOLDEN_PATH}. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if not data:
        print(f"❌ {GOLDEN_PATH} rỗng. Hãy tạo ít nhất 1 test case.")
        return None
    return data


async def run_version(version: str, dataset, retriever: Retriever):
    """Chạy benchmark cho 1 phiên bản agent, trả (results, metrics, tracker)."""
    print(f"🚀 Benchmark {version} ({len(dataset)} cases)...")
    tracker = CostTracker()
    agent = MainAgent(version=version, retriever=retriever)
    runner = BenchmarkRunner(agent, judge=MultiJudge(), tracker=tracker)
    t0 = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - t0
    metrics = report.aggregate_metrics(results)
    metrics["wall_clock_s"] = round(elapsed, 2)
    print(f"   ✔ {version}: avg_score={metrics['avg_score']}, "
          f"hit_rate={metrics['hit_rate']}, pass_rate={metrics['pass_rate']}, {elapsed:.1f}s")
    return results, metrics, tracker


async def position_bias_probe(v2_results, sample=5):
    """Đo position bias trên một mẫu để chứng minh độ tin cậy của Judge."""
    judge = MultiJudge()
    tracker = CostTracker()
    sample_cases = v2_results[:sample]
    checks = await asyncio.gather(*[
        judge.check_position_bias(r["question"], r["agent_answer"], r["expected_answer"], tracker)
        for r in sample_cases
    ])
    consistent = sum(1 for c in checks if c["consistent"])
    return {
        "samples": len(checks),
        "consistent": consistent,
        "consistency_rate": round(consistent / len(checks), 3) if checks else 0.0,
        "note": "Tỉ lệ judge giữ nguyên phán đoán khi đảo vị trí A/B. Càng cao càng ít thiên vị vị trí.",
    }


def release_gate(v1_m, v2_m, v1_cost, v2_cost):
    """Tự động quyết định Release/Block dựa trên Chất lượng & Chi phí."""
    delta = {
        "avg_score": round(v2_m["avg_score"] - v1_m["avg_score"], 3),
        "hit_rate": round(v2_m["hit_rate"] - v1_m["hit_rate"], 3),
        "faithfulness": round(v2_m["faithfulness"] - v1_m["faithfulness"], 3),
        "pass_rate": round(v2_m["pass_rate"] - v1_m["pass_rate"], 3),
    }
    reasons = []
    ok = True

    # Quyết định chất lượng có tính đến NHIỄU của LLM judge:
    # chỉ coi là regression khi điểm tụt VƯỢT vùng dung sai; trong vùng nhiễu thì
    # dùng pass_rate (ổn định hơn) làm tiêu chí phụ.
    if delta["avg_score"] < -GATE_SCORE_TOLERANCE:
        ok = False
        reasons.append(f"Chất lượng giảm ĐÁNG KỂ (Δscore={delta['avg_score']:+.3f} < -{GATE_SCORE_TOLERANCE}).")
    elif delta["avg_score"] < 0:
        if delta["pass_rate"] < 0:
            ok = False
            reasons.append(f"Δscore trong vùng nhiễu nhưng pass_rate giảm ({delta['pass_rate']:+.3f}) → coi là regression.")
        else:
            reasons.append(f"Δscore giảm nhẹ trong vùng nhiễu (|{delta['avg_score']:+.3f}| ≤ {GATE_SCORE_TOLERANCE}), "
                           f"pass_rate không giảm ({delta['pass_rate']:+.3f}) → coi như không đổi.")
    else:
        reasons.append(f"Chất lượng tăng/giữ (Δscore={delta['avg_score']:+.3f}).")

    if v2_m["faithfulness"] < GATE_MIN_FAITHFULNESS:
        ok = False
        reasons.append(f"Faithfulness {v2_m['faithfulness']:.2f} < ngưỡng {GATE_MIN_FAITHFULNESS}.")
    else:
        reasons.append(f"Faithfulness {v2_m['faithfulness']:.2f} đạt ngưỡng.")

    v1_cpe = v1_cost["cost_per_eval_usd"] or 1e-9
    cost_ratio = (v2_cost["cost_per_eval_usd"] / v1_cpe) if v1_cpe else 1.0
    if cost_ratio > GATE_MAX_COST_RATIO:
        ok = False
        reasons.append(f"Chi phí/eval tăng quá mức (x{cost_ratio:.2f} > {GATE_MAX_COST_RATIO}).")
    else:
        reasons.append(f"Chi phí/eval trong ngưỡng (x{cost_ratio:.2f}).")

    return {
        "delta": delta,
        "cost_ratio_v2_over_v1": round(cost_ratio, 3),
        "decision": "APPROVE" if ok else "BLOCK",
        "reasons": reasons,
    }


async def main():
    print(config.mode_banner())
    dataset = load_dataset()
    if dataset is None:
        return

    # Chia sẻ 1 retriever (embed corpus 1 lần) cho cả V1 và V2 để tiết kiệm chi phí
    retriever = Retriever()
    embed_tracker = CostTracker()
    await retriever.build_index(tracker=embed_tracker)

    v1_results, v1_m, v1_tracker = await run_version("V1", dataset, retriever)
    v2_results, v2_m, v2_tracker = await run_version("V2", dataset, retriever)

    print("\n🔬 Kiểm tra Position Bias (mẫu)...")
    pbias = await position_bias_probe(v2_results)

    n = len(dataset)
    v1_cost = v1_tracker.summary(n)
    v2_cost = v2_tracker.summary(n)

    gate = release_gate(v1_m, v2_m, v1_cost, v2_cost)

    print("\n📊 --- REGRESSION V1 vs V2 ---")
    print(f"V1 avg_score: {v1_m['avg_score']}  |  V2 avg_score: {v2_m['avg_score']}")
    print(f"Δ score: {gate['delta']['avg_score']:+.3f} | Δ hit_rate: {gate['delta']['hit_rate']:+.3f}")
    print(f"Chi phí/eval: V1=${v1_cost['cost_per_eval_usd']} | V2=${v2_cost['cost_per_eval_usd']}")
    print(f"🚦 QUYẾT ĐỊNH: {gate['decision']}")
    for rsn in gate["reasons"]:
        print(f"   - {rsn}")

    # ---- Ghi reports ----
    summary = {
        "metadata": {
            "version": "Agent_V2_Optimized",
            "total": n,
            "mode": "MOCK" if config.MOCK_MODE else "LIVE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": v2_m,                 # metrics của bản được release (V2)
        "cost": v2_cost,
        "embedding_cost": embed_tracker.summary(n),
        "position_bias": pbias,
        "regression": {
            "v1_metrics": v1_m,
            "v2_metrics": v2_m,
            "v1_cost": v1_cost,
            "v2_cost": v2_cost,
            **gate,
        },
        "cost_optimization_proposal": {
            "idea": "Giảm ~30% chi phí eval mà không giảm độ chính xác.",
            "tactics": [
                "Dùng gpt-4o-mini làm judge mặc định, chỉ escalate gpt-4o khi 2 judge xung đột (cascade).",
                "Cache embedding corpus (đã làm) và cache kết quả judge cho câu trả lời trùng lặp.",
                "Gộp faithfulness + relevancy vào 1 lời gọi LLM thay vì 2.",
            ],
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"v1": v1_results, "v2": v2_results}, f, ensure_ascii=False, indent=2)

    # ---- Sinh báo cáo phân tích lỗi ----
    os.makedirs("analysis", exist_ok=True)
    md = report.render_failure_analysis(v2_results, v2_m, gate)
    with open("analysis/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n✅ Đã ghi: reports/summary.json, reports/benchmark_results.json, analysis/failure_analysis.md")


if __name__ == "__main__":
    asyncio.run(main())
