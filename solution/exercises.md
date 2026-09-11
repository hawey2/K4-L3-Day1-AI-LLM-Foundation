# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature càng cao thì mức độ ngẫu nhiên và sáng tạo của đầu ra càng tăng.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> 0.2–0.3. Mục tiêu của hệ thống là cung cấp thông tin chính xác, nhất quán và đáng tin cậy thay vì sáng tạo nội dung mới. Điều này đảm bảo sự ổn định cho người dùng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn khoảng 16,7 lần so với GPT-4o-mini cho cùng một lượng token đầu ra. GPT-4o phù hợp khi cần chất lượng suy luận cao, chẳng hạn trợ lý nghiên cứu, phân tích tài liệu pháp lý hoặc hỗ trợ ra quyết định phức tạp. Ngược lại, GPT-4o-mini phù hợp cho chatbot FAQ, hỗ trợ khách hàng cơ bản hoặc các ứng dụng có lưu lượng lớn và yêu cầu tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi có sự khác biệt rõ rệt về cách diễn đạt và mức độ chi tiết. Với persona "giáo viên tiểu học", câu trả lời thường ngắn gọn, đơn giản. Trong khi đó, persona "chuyên gia tài chính" sử dụng nhiều thuật ngữ kỹ thuật như sổ cái phân tán (distributed ledger),.. để giải thích sâu hơn về nguyên lý hoạt động. Điều này cho thấy system prompt đóng vai trò định hướng hành vi của mô hình, ảnh hưởng trực tiếp đến giọng điệu, độ sâu kiến thức, cách lựa chọn từ vựng và đối tượng người đọc mà câu trả lời hướng tới. 

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khi thử với một đoạn văn tiếng Việt khoảng 100 từ, phương pháp ước lượng theo công thức số từ / 0.75 cho kết quả khoảng 133 token. Trong khi đó, count_tokens() sử dụng tiktoken có thể cho kết quả khoảng 145–160 token tùy nội dung cụ thể. Như vậy sai lệch thường nằm trong khoảng 10–20%.

> Hệ thống token hóa được tối ưu chủ yếu trên dữ liệu tiếng Anh. Các dấu thanh, ký tự Unicode và nhiều từ ghép tiếng Việt thường bị tách thành nhiều token nhỏ hơn so với các từ tiếng Anh phổ biến đã xuất hiện nhiều trong dữ liệu huấn luyện. Do đó cùng một lượng thông tin, văn bản tiếng Việt thường sử dụng nhiều token hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài hoặc thời gian xử lý của mô hình kéo dài vài giây trở lên, chẳng hạn chatbot hỗ trợ khách hàng, trợ lý lập trình hoặc hệ thống phân tích tài liệu. Người dùng có thể nhìn thấy câu trả lời xuất hiện ngay lập tức thay vì phải chờ toàn bộ kết quả, từ đó giảm cảm giác chờ đợi và tăng trải nghiệm tương tác. Ngược lại, non-streaming phù hợp hơn với các tác vụ ngắn, yêu cầu xử lý hoàn chỉnh trước khi hiển thị hoặc khi hệ thống cần kiểm tra, định dạng và kiểm duyệt toàn bộ nội dung trước khi gửi cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm tải cho hệ thống khi API đang quá tải bằng cách tăng dần thời gian chờ sau mỗi lần thất bại. Nếu sử dụng delay cố định, hàng nghìn client có thể đồng thời gửi lại yêu cầu sau đúng 1 giây, tạo ra một "đợt sóng" truy cập mới và khiến hệ thống tiếp tục quá tải. Với exponential backoff, các lần retry được giãn cách ngày càng xa nhau, giúp máy chủ có thời gian phục hồi và tăng xác suất yêu cầu thành công ở những lần thử sau. Đây là cơ chế phổ biến trong các hệ thống phân tán và dịch vụ đám mây.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> "Bạn là trợ giảng AI thân thiện của khóa học AI. Hãy trả lời ngắn gọn, rõ ràng bằng tiếng Việt, ưu tiên giải thích theo từng bước và đưa ví dụ thực tế khi cần."

> Tôi chọn cụm từ "trả lời ngắn gọn" để tránh phản hồi quá dài, giúp người học dễ tập trung vào ý chính. Cụm từ "bằng tiếng Việt" đảm bảo tính nhất quán trong toàn bộ cuộc hội thoại và phù hợp với đối tượng học viên của khóa học. Ngoài ra, yêu cầu "giải thích theo từng bước" giúp các khái niệm kỹ thuật trở nên dễ hiểu hơn đối với người mới bắt đầu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> chỉ lưu tối đa 3 lượt hội thoại gần nhất (6 messages), do đó có thể quên các thông tin quan trọng đã được đề cập từ đầu cuộc trò chuyện. Điều này làm giảm khả năng duy trì ngữ cảnh trong các phiên trao đổi dài.

> Một cải thiện khả thi là bổ sung cơ chế conversation summarization. Khi lịch sử hội thoại vượt quá giới hạn, hệ thống sẽ sử dụng mô hình để tạo một bản tóm tắt ngắn gọn của các nội dung quan trọng và lưu bản tóm tắt này vào system prompt hoặc memory riêng. Nhờ đó chatbot vẫn duy trì được ngữ cảnh dài hạn mà không làm tăng quá nhiều số lượng token gửi đến API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
