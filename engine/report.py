"""
Tổng hợp số liệu, phân cụm lỗi (failure clustering) và sinh báo cáo.
"""
from typing import List, Dict
from statistics import mean

from engine import config
from engine.llm_judge import cohen_kappa
from engine.retrieval_eval import RetrievalEvaluator

_ret = RetrievalEvaluator()


def aggregate_metrics(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {}

    scores_a = [r["judge"]["individual_scores"].get(config.JUDGE_A_MODEL) for r in results]
    scores_b = [r["judge"]["individual_scores"].get(config.JUDGE_B_MODEL) for r in results]
    # lọc None (phòng trường hợp model name khác)
    pairs = [(a, b) for a, b in zip(scores_a, scores_b) if a is not None and b is not None]
    kappa = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else 0.0

    ret_agg = _ret.aggregate([r["retrieval"] for r in results])

    return {
        "avg_score": round(mean(r["judge"]["final_score"] for r in results), 3),
        "pass_rate": round(sum(1 for r in results if r["status"] == "pass") / n, 3),
        "hit_rate": ret_agg["avg_hit_rate"],
        "mrr": ret_agg["avg_mrr"],
        "retrieval_evaluated_cases": ret_agg["evaluated_cases"],
        "faithfulness": round(mean(r["faithfulness"] for r in results), 3),
        "relevancy": round(mean(r["relevancy"] for r in results), 3),
        "agreement_rate": round(mean(r["judge"]["agreement"] for r in results), 3),
        "cohen_kappa": kappa,
        "conflicts": sum(1 for r in results if r["judge"]["conflict"]),
        "avg_latency_s": round(mean(r["latency_s"] for r in results), 3),
        "max_latency_s": round(max(r["latency_s"] for r in results), 3),
    }


def cluster_failures(results: List[Dict]) -> Dict[str, Dict]:
    """Phân cụm các case fail theo nguyên nhân (failure mode)."""
    clusters: Dict[str, Dict] = {}

    def add(name, r, cause):
        c = clusters.setdefault(name, {"count": 0, "cases": [], "likely_cause": cause})
        c["count"] += 1
        if len(c["cases"]) < 5:
            c["cases"].append(r["id"])

    for r in results:
        if r["status"] == "pass":
            continue
        ftype = r["type"]
        if r["faithfulness"] < 0.5:
            add("Hallucination", r, "Agent bịa thông tin / context không hỗ trợ câu trả lời")
        if r["retrieval"]["hit_rate"] == 0.0:
            add("Retrieval Miss", r, "Vector search không lấy được tài liệu đúng trong top-k")
        if ftype in ("prompt_injection", "goal_hijacking"):
            add("Security Breach", r, "Agent bị lừa làm việc ngoài phạm vi / lộ thông tin")
        if r["relevancy"] < 0.5:
            add("Off-topic / Incomplete", r, "Câu trả lời lạc đề hoặc thiếu so với câu hỏi")
        if ftype in ("out_of_context", "hallucination_trap") and r["faithfulness"] < 0.7:
            add("Failed-to-Refuse", r, "Agent bịa/không nói 'không biết' khi context thiếu thông tin")

    return dict(sorted(clusters.items(), key=lambda kv: -kv[1]["count"]))


def worst_cases(results: List[Dict], k: int = 3) -> List[Dict]:
    return sorted(results, key=lambda r: r["judge"]["final_score"])[:k]


def render_failure_analysis(v2_results: List[Dict], v2_metrics: Dict, regression: Dict) -> str:
    n = len(v2_results)
    n_pass = sum(1 for r in v2_results if r["status"] == "pass")
    clusters = cluster_failures(v2_results)
    worst = worst_cases(v2_results, 3)

    lines = []
    lines.append("# Báo cáo Phân tích Thất bại (Failure Analysis Report)\n")
    lines.append("> File này được sinh tự động từ kết quả benchmark thật (`reports/benchmark_results.json`).\n")

    lines.append("## 1. Tổng quan Benchmark")
    lines.append(f"- **Tổng số cases:** {n}")
    lines.append(f"- **Tỉ lệ Pass/Fail:** {n_pass}/{n - n_pass}  (pass_rate = {v2_metrics['pass_rate']*100:.1f}%)")
    lines.append("- **Điểm chất lượng sinh (custom RAGAS):**")
    lines.append(f"    - Faithfulness: {v2_metrics['faithfulness']:.2f}")
    lines.append(f"    - Relevancy: {v2_metrics['relevancy']:.2f}")
    lines.append(f"- **Retrieval:** Hit Rate = {v2_metrics['hit_rate']*100:.1f}% | MRR = {v2_metrics['mrr']:.3f}")
    lines.append(f"- **Điểm Multi-Judge trung bình:** {v2_metrics['avg_score']:.2f} / 5.0")
    lines.append(f"- **Độ tin cậy giám khảo:** Agreement = {v2_metrics['agreement_rate']*100:.1f}% | "
                 f"Cohen's Kappa = {v2_metrics['cohen_kappa']:.3f} | Xung đột = {v2_metrics['conflicts']} case\n")

    lines.append("## 2. Phân nhóm lỗi (Failure Clustering)")
    if clusters:
        lines.append("| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến | Case ví dụ |")
        lines.append("|----------|----------|---------------------|------------|")
        for name, c in clusters.items():
            lines.append(f"| {name} | {c['count']} | {c['likely_cause']} | {', '.join(c['cases'])} |")
    else:
        lines.append("_Không có case fail nào — toàn bộ đạt._")
    lines.append("")

    lines.append("## 3. Phân tích 5 Whys (3 case điểm thấp nhất)\n")
    for i, r in enumerate(worst, 1):
        lines.append(f"### Case #{i}: `{r['id']}` ({r['type']}, {r['difficulty']}) — điểm {r['judge']['final_score']}")
        lines.append(f"- **Câu hỏi:** {r['question']}")
        lines.append(f"- **Agent trả lời:** {r['agent_answer'][:200]}")
        lines.append(f"- **Symptom:** điểm thấp ({r['judge']['final_score']}/5), "
                     f"faithfulness={r['faithfulness']}, hit_rate={r['retrieval']['hit_rate']}.")
        lines.append(f"- **Why 1:** {_why1(r)}")
        lines.append(f"- **Why 2:** {_why2(r)}")
        lines.append("- **Why 3:** Chiến lược chunking/retrieval hoặc prompt chưa phù hợp với loại câu hỏi này.")
        lines.append(f"- **Root Cause (đề xuất):** {_root_cause(r)}\n")

    lines.append("## 4. Kế hoạch cải tiến (Action Plan)")
    lines.append("- [ ] Tăng chất lượng retrieval (reranking / điều chỉnh top_k / semantic chunking).")
    lines.append("- [ ] Củng cố system prompt: bắt buộc 'chỉ trả lời theo context' và nói 'không biết' khi thiếu dữ liệu.")
    lines.append("- [ ] Thêm guardrail chống prompt-injection cho các case red-team.")
    lines.append("- [ ] Bổ sung tài liệu cho các chủ đề người dùng hỏi nhưng corpus chưa có.\n")

    lines.append("## 5. Ghi chú Regression")
    d = regression["delta"]
    lines.append(f"- Δ avg_score (V2 - V1): {d['avg_score']:+.3f}")
    lines.append(f"- Δ hit_rate: {d['hit_rate']:+.3f} | Δ faithfulness: {d['faithfulness']:+.3f}")
    lines.append(f"- Quyết định Release Gate: **{regression['decision']}**")
    return "\n".join(lines) + "\n"


def _why1(r):
    if r["retrieval"]["hit_rate"] == 0.0:
        return "Retriever không lấy được tài liệu chứa câu trả lời (sai context ngay từ đầu)."
    if r["type"] in ("prompt_injection", "goal_hijacking"):
        return "Agent bị câu hỏi điều khiển, đi chệch nhiệm vụ hỗ trợ IT."
    if r["faithfulness"] < 0.5:
        return "Agent đưa thông tin không có trong context (hallucinate)."
    return "Câu trả lời chưa khớp với đáp án chuẩn về nội dung hoặc mức độ đầy đủ."


def _why2(r):
    if r["retrieval"]["hit_rate"] == 0.0:
        return "Embedding/cosine xếp tài liệu đúng ngoài top_k, hoặc corpus thiếu thông tin."
    if r["type"] == "out_of_context":
        return "Prompt chưa đủ mạnh để buộc agent từ chối khi không có dữ liệu."
    return "System prompt chưa ràng buộc agent bám sát context và đúng trọng tâm."


def _root_cause(r):
    if r["retrieval"]["hit_rate"] == 0.0:
        return "Ingestion/Chunking & Retrieval — cần semantic chunking + reranking."
    if r["type"] in ("prompt_injection", "goal_hijacking", "out_of_context"):
        return "Prompting/Guardrail — thiếu lớp phòng thủ và quy tắc từ chối."
    return "Prompting — cần rubric chặt hơn về tính bám context."
