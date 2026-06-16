# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> File này được sinh tự động từ kết quả benchmark thật (`reports/benchmark_results.json`).

## 1. Tổng quan Benchmark
- **Tổng số cases:** 55
- **Tỉ lệ Pass/Fail:** 55/0  (pass_rate = 100.0%)
- **Điểm chất lượng sinh (custom RAGAS):**
    - Faithfulness: 1.00
    - Relevancy: 0.86
- **Retrieval:** Hit Rate = 100.0% | MRR = 0.980
- **Điểm Multi-Judge trung bình:** 4.84 / 5.0
- **Độ tin cậy giám khảo:** Agreement = 92.7% | Cohen's Kappa = 0.427 | Xung đột = 0 case

## 2. Phân nhóm lỗi (Failure Clustering)
_Không có case fail nào — toàn bộ đạt._

## 3. Phân tích 5 Whys (3 case điểm thấp nhất)

### Case #1: `GS-002` (fact-check, medium) — điểm 3.0
- **Câu hỏi:** Mật khẩu mới của tôi cần đáp ứng những yêu cầu gì?
- **Agent trả lời:** Mật khẩu mới của bạn cần đáp ứng các yêu cầu sau:

1. **Độ dài tối thiểu**: Mật khẩu phải dài ít nhất 12 ký tự.
2. **Phân loại ký tự**: Mật khẩu phải bao gồm:
   - Chữ hoa
   - Chữ thường
   - Số
   -
- **Symptom:** điểm thấp (3.0/5), faithfulness=1.0, hit_rate=1.0.
- **Why 1:** Câu trả lời chưa khớp với đáp án chuẩn về nội dung hoặc mức độ đầy đủ.
- **Why 2:** System prompt chưa ràng buộc agent bám sát context và đúng trọng tâm.
- **Why 3:** Chiến lược chunking/retrieval hoặc prompt chưa phù hợp với loại câu hỏi này.
- **Root Cause (đề xuất):** Prompting — cần rubric chặt hơn về tính bám context.

### Case #2: `GS-010` (fact-check, easy) — điểm 4.0
- **Câu hỏi:** Làm thế nào để thêm tài khoản email công ty trên điện thoại?
- **Agent trả lời:** Để thêm tài khoản email công ty trên điện thoại, bạn cần thực hiện các bước sau:

1. **Chọn loại tài khoản**: Trên điện thoại, chọn thêm tài khoản loại 'Exchange'.
2. **Nhập thông tin tài khoản**: Nhậ
- **Symptom:** điểm thấp (4.0/5), faithfulness=1.0, hit_rate=1.0.
- **Why 1:** Câu trả lời chưa khớp với đáp án chuẩn về nội dung hoặc mức độ đầy đủ.
- **Why 2:** System prompt chưa ràng buộc agent bám sát context và đúng trọng tâm.
- **Why 3:** Chiến lược chunking/retrieval hoặc prompt chưa phù hợp với loại câu hỏi này.
- **Root Cause (đề xuất):** Prompting — cần rubric chặt hơn về tính bám context.

### Case #3: `GS-028` (fact-check, easy) — điểm 4.0
- **Câu hỏi:** Tôi cần làm gì trước khi sử dụng Remote Desktop để làm việc từ xa?
- **Agent trả lời:** Trước khi sử dụng Remote Desktop để làm việc từ xa, bạn cần thực hiện các bước sau:

1. **Kết nối VPN**: Trước tiên, bạn phải kết nối với VPN. Công ty sử dụng VPN client GlobalConnect, bạn có thể tải 
- **Symptom:** điểm thấp (4.0/5), faithfulness=1.0, hit_rate=1.0.
- **Why 1:** Câu trả lời chưa khớp với đáp án chuẩn về nội dung hoặc mức độ đầy đủ.
- **Why 2:** System prompt chưa ràng buộc agent bám sát context và đúng trọng tâm.
- **Why 3:** Chiến lược chunking/retrieval hoặc prompt chưa phù hợp với loại câu hỏi này.
- **Root Cause (đề xuất):** Prompting — cần rubric chặt hơn về tính bám context.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Tăng chất lượng retrieval (reranking / điều chỉnh top_k / semantic chunking).
- [ ] Củng cố system prompt: bắt buộc 'chỉ trả lời theo context' và nói 'không biết' khi thiếu dữ liệu.
- [ ] Thêm guardrail chống prompt-injection cho các case red-team.
- [ ] Bổ sung tài liệu cho các chủ đề người dùng hỏi nhưng corpus chưa có.

## 5. Ghi chú Regression
- Δ avg_score (V2 - V1): +0.118
- Δ hit_rate: +0.000 | Δ faithfulness: +0.018
- Quyết định Release Gate: **APPROVE**
