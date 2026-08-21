# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Fine-tune tăng target từ 0.765 lên 0.965 và vẫn cho JSON hợp lệ 100%, nhưng vẫn bị verdict FAILED vì regression giảm 0.080. Trước khi đo, tôi dễ coi target cao là đủ; regression gate cho thấy đó là một kết luận thiếu điều kiện.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần lâu nhất là bốn lượt sinh để đóng băng baseline, train các contrast và chấm lại trên target. Tôi dự đoán training sẽ là nút thắt, nhưng generation/evaluation cũng tốn đáng kể vì mỗi adapter phải được nạp và sinh lại. Việc giữ adapter đã xong để resume giúp tránh mất toàn bộ thời gian khi runtime Colab bị ngắt.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi không còn tin final training loss thấp hơn đồng nghĩa mô hình tốt hơn hoặc deploy được. `attn_only` có loss thấp và target nhỉnh hơn trong task hẹp, trong khi `wrong_lr` cho thấy chỉ đổi LR có thể làm target về 0; còn verdict cuối cùng lại do regression quyết định.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc cấu trúc artefact, diễn giải gatekeeper và tổ chức report theo số liệu JSON/CSV. Nó không thể thay thế kết quả GPU: các dự đoán mẫu của baseline (b) không được persist, nên không thể tạo lại trung thực sau khi đã đóng băng. Tôi chỉ dùng số liệu đã có trong artefact.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên là gì?**

Tôi sẽ định nghĩa task metric, tập regression và prompt baseline mạnh trước khi thu thập/train dữ liệu. Sau đó kiểm tra mask bằng token span, đóng băng eval, rồi mới chạy training. Điều này ngăn việc tối ưu nhầm theo loss hoặc chỉnh prompt sau khi nhìn thấy kết quả.
