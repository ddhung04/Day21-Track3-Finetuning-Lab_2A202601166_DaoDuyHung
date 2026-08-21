# Lab 21 — Evaluation Report (tạm thời)

**Họ tên**: Dao Duy Hung
**MSSV**: 2A202601166
**Ngày**: 2026-08-21
**Tier**: `CPU`
**Base model**: `Qwen/Qwen3.5-0.8B`
**GPU thực tế**: Không có GPU

> Báo cáo này chỉ ghi các kết quả đã đo được trên đường CPU. NB2–NB5 chưa chạy vì môi trường hiện tại không có PyTorch/GPU; không có số liệu baseline, LoRA hay verdict để báo cáo trung thực.

## 1. Setup

| Hạng mục | Kết quả |
|---|---|
| Dataset | 250 ticket CSKH tổng hợp → JSON triage 4 trường |
| Train / val | 225 / 25, seed 42 |
| `max_length` | 512; p95 đo được là 98, `suggested_max_length` là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch được cấu hình; chưa có `max_steps` vì chưa train |

Template giữ khối `<think>`: có. `template_check.json` kết luận reasoning được giữ lại, nên template không tự xóa trace khi render.

## 2. Mask proof (NB1)

| Hạng mục | Kết quả |
|---|---|
| `supervised_fraction` | 0.3936 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đoạn đầu được tính loss:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Mask `assistant-only` chỉ supervise câu trả lời JSON và token kết thúc; ticket của người dùng không xuất hiện trong span loss. So với mode `everything` (94/94 token, 100%), tỷ lệ 37/94 token (39.36%) là bằng chứng trực tiếp rằng prompt không bị đưa vào loss.

## 3. Ba baseline (NB2 — đo trước khi train)

Chưa chạy được. `baselines_frozen.json` chưa tồn tại vì môi trường hiện tại không cài PyTorch và không có GPU để tải/sinh từ base model. Không suy diễn hay điền số liệu thay thế.

## 4. Giải phẫu cấu hình sai (NB4)

Chưa chạy được. Chưa có `adapters/correct/`, `runs.csv` hoặc `autopsy.json`; vì vậy không thể xếp hạng `correct`, `attn_only`, `wrong_lr`, và `qlora` theo target score. Khi có GPU, các run sẽ dùng cùng step budget; `attn_only` phải được tính matched rank để giữ chênh lệch trainable parameters dưới 5%.

## 5. Phán quyết (NB5)

Chưa có phán quyết. Cổng regression chỉ được chạy sau khi baseline (b) đã được đóng băng và adapter `correct` đã được chấm trên đủ tập target/regression. Kết quả hợp lệ có thể là PASSED hoặc FAILED; hiện chưa có dữ liệu để kết luận deploy.

## 6. Định tính

Chưa có `qualitative.json`, do chưa có dự đoán fine-tune. Sau khi hoàn tất NB5, báo cáo sẽ chọn ít nhất năm mẫu và bắt buộc nêu ít nhất hai trường hợp fine-tune thua để tránh cherry-picking.

## 7. Kết luận & điều tôi học được

Kết quả hiện có xác nhận nền tảng dữ liệu của pipeline hoạt động đúng trước khi đầu tư thời gian GPU. Mask `assistant-only` supervise 39.36% token, chứa JSON mục tiêu và loại phần ticket ra khỏi loss; đây là điều kiện cần để mô hình không học lặp lại prompt. Chat template cũng giữ `<think>`, nên nếu dataset sau này có reasoning trace thì lựa chọn mask sẽ có ý nghĩa thực nghiệm thay vì bị template âm thầm xóa. Độ dài p95 chỉ là 98 token, trong khi tier CPU dùng `max_length=512`; khoảng cách này cần được ghi nhận như một trade-off cấu hình, không che giấu bằng cách sửa số đo.

Tuy nhiên, chưa thể suy ra LoRA có đáng deploy hay không. Điều đó cần baseline optimized prompt được đo trước train, adapter được huấn luyện, và regression gate được chấm trên tập eval chưa thay đổi. Khi chưa có các phép đo này, một tuyên bố rằng fine-tune thắng hay thua đều không có cơ sở. Bước tiếp theo là chạy cùng một pipeline NB1–NB5 trên Colab T4, để `EVAL_LIMIT` trống, lưu đầy đủ artefact rồi cập nhật báo cáo bằng các số liệu thực tế.

Ba điều tôi học được:

1. Mask phải được chứng minh bằng span token có thể đọc được, không chỉ tin vào một cờ thư viện.
2. Prompt mạnh phải được đóng băng trước khi train; so với baseline yếu là một phép so sánh không trung thực.
3. Train loss không thay thế target score; chỉ đánh giá trên task và regression gate mới hỗ trợ quyết định deploy.

Nếu có thêm thời gian, tôi sẽ chạy full pipeline trên T4, so sánh bốn cấu hình bằng target score, rồi cập nhật phần định tính bằng cả ca thắng và ca thua.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
