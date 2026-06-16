# Báo cáo Cá nhân (Individual Reflection)

> Bám theo 3 tiêu chí chấm điểm cá nhân: Engineering Contribution, Technical Depth, Problem Solving.

**Họ tên:** Lưu Xuân Thế  
**Mã sinh viên:** 2A202600983  
**Vai trò:** Thực hiện cá nhân (toàn bộ hệ thống)

---

## 1. Đóng góp kỹ thuật (Engineering Contribution — 15đ)
Liệt kê cụ thể module/file bạn làm và dẫn chứng qua Git commits.

| Module / File | Bạn đã làm gì | Commit liên quan |
|---------------|---------------|------------------|
| `engine/llm_judge.py` | (ví dụ) Viết logic Multi-Judge & xử lý xung đột | `<hash>` |
| `engine/runner.py` | (ví dụ) Async + Semaphore giới hạn concurrency | `<hash>` |
| ... | ... | ... |

---

## 2. Chiều sâu kỹ thuật (Technical Depth — 15đ)
Giải thích bằng lời của bạn (chấm điểm dựa trên mức độ hiểu thật):

- **MRR (Mean Reciprocal Rank):** ____________________________________________
  - (Gợi ý: nghịch đảo thứ hạng của tài liệu đúng đầu tiên; vị trí 1 → 1.0, vị trí 2 → 0.5...
    Khác Hit Rate ở chỗ MRR quan tâm *thứ hạng*, Hit Rate chỉ quan tâm *có/không* trong top-k.)
- **Cohen's Kappa:** _________________________________________________________
  - (Gợi ý: đo độ đồng thuận giữa 2 giám khảo *sau khi trừ đi phần đồng thuận do may rủi*.
    kappa = (Po − Pe)/(1 − Pe). Vì sao tốt hơn agreement thô?)
- **Position Bias:** ________________________________________________________
  - (Gợi ý: LLM-judge có xu hướng thiên vị câu trả lời ở một vị trí nhất định;
    ta kiểm tra bằng cách đảo chỗ A/B và xem phán đoán có đổi không.)
- **Trade-off Chi phí ↔ Chất lượng:** ______________________________________
  - (Gợi ý: dùng model rẻ làm judge mặc định, chỉ escalate model mạnh khi xung đột — xem
    `cost_optimization_proposal` trong `reports/summary.json`.)

---

## 3. Giải quyết vấn đề (Problem Solving — 10đ)
Một vấn đề khó bạn gặp khi xây hệ thống và cách bạn xử lý:

- **Vấn đề:** _______________________________________________________________
- **Cách chẩn đoán:** _______________________________________________________
- **Giải pháp & kết quả:** __________________________________________________

---

## 4. Bài học rút ra
_______________________________________________________________________________
