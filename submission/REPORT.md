# Lab 21 — Evaluation Report

**Họ tên**: Dao Duy Hung
**MSSV**: 2A202601166
**Ngày**: 2026-08-21
**Tier**: `T4`
**Base model**: `unsloth/Qwen3.5-4B`
**GPU thực tế**: Tesla T4 (16 GB)

## 1. Setup

| Hạng mục | Kết quả |
|---|---|
| Dataset | 250 ticket CSKH tổng hợp → JSON triage 4 trường |
| Train / val | 225 / 25, seed 42 |
| `max_length` | 1024; p95 đo được là 98 (`suggested_max_length=256`) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch / 30 step cho cả bốn run |

`max_length=1024` là giá trị tier T4; p95 98 cho thấy corpus ngắn hơn nhiều. Tôi giữ cấu hình tier thay vì sửa số đo để phép so sánh giữa các run có cùng giới hạn chuỗi.

**Template có giữ `<think>` không?** Có. `template_check.json` xác nhận cả open tag và body còn sau khi render: `reasoning preserved — safe to train on traces`.

## 2. Mask proof (NB1)

| Hạng mục | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149 (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Tỷ lệ 41.49% thấp hơn rất xa 95%, đồng thời preview không chứa ticket của user. Vì vậy loss đang học câu trả lời JSON chứ không học cách lặp lại prompt.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3172.7 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1051.2 |
| (c) LoRA fine-tune | 0.965 | 0.6778 | 1.000 | 1412.0 |

Baseline (b) mạnh hơn (a) rõ rệt ở target (+0.765) và format (+1.000), nên đây là mốc hợp lệ để so sánh. Tôi không sửa `OPTIMIZED_PROMPT` sau khi đóng băng; SHA được lưu là `719e74d3b6232053`. Fine-tune cải thiện target thêm +0.200 so với (b), nhưng latency tăng 360.8 ms/mẫu và regression giảm 0.0800.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target | format | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6283 | 0.965 | 1.000 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5372 | 0.970 | 1.000 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 0.000 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1.000 | 7.09 |

Tất cả các run đều có `max_steps=30`. `attn_only` có 32,456,704 tham số, lệch 0.025% so với `correct`, nên đây là contrast công bằng.

**4.1 — Vị trí và rank.** Với cùng ngân sách tham số, `attn_only` đạt target 0.970, cao hơn rất nhẹ `correct` (0.965). Train loss của nó cũng thấp hơn (0.5372 so với 0.6283), nên ở lần đo này thứ tự target và loss trùng nhau. Tuy nhiên, khác biệt target chỉ 0.005 trên tác vụ JSON hẹp; nó không chứng minh rank 283 tự thân là đòn bẩy phổ quát. Nó chỉ cho thấy q,v với rank đã match ngân sách là đủ mạnh cho tác vụ này; cần đo thêm miền khác trước khi thay `text-linear` mặc định.

**4.2 — Learning rate sai.** `wrong_lr` chỉ đổi LR từ 1e-4 sang 1e-5, nhưng final loss cao 1.5704 và target/format đều bằng 0.000. Đây là dấu hiệu learning rate ở thang full fine-tuning quá nhỏ cho LoRA trong ngân sách 30 step. Nếu chỉ nhìn một đoạn loss chưa biết LR, có thể kết luận nhầm rằng placement hoặc rank không đủ; biến thay đổi duy nhất ở đây cho thấy nguyên nhân là tốc độ học.

**4.3 — QLoRA.** QLoRA giảm VRAM từ 12.01 xuống 7.09 GB, tiết kiệm 4.92 GB (khoảng 41%). Đổi lại target giảm 0.025 so với `correct`, train loss tăng từ 0.6283 lên 0.7058 và latency tăng từ 1412.0 lên 1762.9 ms. Số đo này ủng hộ khuyến nghị không dùng QLoRA cho dòng Qwen3.5 khi đủ VRAM: T4 vẫn chứa được LoRA 16-bit, trong khi QLoRA không mang lại target hoặc latency tốt hơn.

## 5. Phán quyết (NB5)

**Kết quả regression gate: FAILED.**

`target Δ = +0.200` · `regression Δ = -0.080` · `valid_trace_rate = 0.000`

Fine-tune thắng optimized prompt ở target, từ 0.765 lên 0.965, đồng thời vẫn giữ format 1.000. Tuy nhiên, regression giảm từ 0.7578 xuống 0.6778, tức giảm 0.080, vượt xa tolerance 0.020. Vì thế verdict FAILED là đúng theo cổng hồi quy, không phải vì target yếu. Corpus huấn luyện chỉ gồm triage JSON và không có replay dữ liệu kiến thức phổ thông, nên adapter đã chuyên biệt hóa hành vi theo schema nhưng làm suy giảm một phần năng lực trả lời tổng quát. `valid_trace_rate=0` cũng phù hợp với corpus đáp án JSON không có reasoning trace. Tôi không nên deploy adapter này làm thay thế chung cho base model; nếu dùng cho triage, cần thêm 1–5% replay dữ liệu phổ thông, train lại, và chấm lại trên đúng eval set đã đóng băng.

## 6. Định tính — có cả ca thua

`qualitative.json` lưu dự đoán fine-tune và điểm từng mẫu; prediction từng mẫu của baseline (b) không được persist, nên không thể dựng lại trung thực sau khi đã đóng băng. Bảng dưới đối chiếu nhãn với fine-tune và ghi rõ điểm trường; aggregate baseline (b) ở mục 3 là 0.765.

| # | Ticket rút gọn | Nhãn đúng | Fine-tune | Nhận xét |
|---:|---|---|---|---|
| 0 | Chuột không dây, trả lại, gấp | `doi_tra/cao/chuột không dây/tich_cuc` | Đúng 4/4 (1.00) | ✅ FT thắng nhãn |
| 1 | Ốp lưng điện thoại, hoàn tiền | `hoan_tien/trung_binh/ốp lưng điện thoại/tieu_cuc` | Đúng 4/4 (1.00) | ✅ FT thắng nhãn |
| 3 | Bình giữ nhiệt, chưa thấy tiền, khi nào tiện | `hoan_tien/thap/bình giữ nhiệt/tich_cuc` | `urgency=trung_binh`, còn lại đúng (0.75) | ❌ FT thua ở urgency |
| 5 | Nồi chiên không dầu, thiếu phụ kiện, khi nào tiện | `san_pham_loi/thap/nồi chiên không dầu/trung_tinh` | `urgency=trung_binh`, còn lại đúng (0.75) | ❌ FT thua ở urgency |
| 26 | Máy xay sinh tố, trả lại tiền, không vội | `hoan_tien/thap/máy xay sinh tố/trung_tinh` | `intent=doi_tra`, còn lại đúng (0.75) | ❌ FT nhầm hoàn tiền với đổi trả |

Các ca thua đều là ticket có tín hiệu urgency mềm như “khi nào tiện”, “không vội”, hoặc ranh giới ngữ nghĩa giữa `hoan_tien` và `doi_tra`. Đây là lỗi phân loại cục bộ, không phải lỗi format: cả ba vẫn tạo JSON hợp lệ.

## 7. Kết luận & điều tôi học được

Thí nghiệm cho thấy LoRA có thể cải thiện rất mạnh đúng tác vụ triage, nhưng kết quả đó chưa đủ để biện minh cho việc deploy như một adapter đa dụng. Nguyên nhân đầu tiên là mask đã đúng: 39/94 token được supervise, JSON mục tiêu nằm trong loss và ticket bị mask, nên mô hình học hành vi cần thiết thay vì copy prompt. Nguyên nhân thứ hai là optimized prompt đã tạo mốc cao 0.765 trước khi train; vì vậy mức 0.965 của fine-tune là cải thiện thật trên target, không phải lợi thế do baseline yếu. Nguyên nhân thứ ba lại dẫn đến verdict FAILED: dữ liệu train chỉ chuyên cho JSON triage, không có replay kiến thức phổ thông, nên regression giảm 0.080 và vượt ngưỡng cho phép 0.020.

Vì vậy tôi chưa deploy adapter này làm mặc định cho mọi yêu cầu. Nó phù hợp hơn như một route chuyên biệt cho ticket triage, hoặc như một checkpoint để train lại với 1–5% replay rồi chấm lại regression. Đòn bẩy nổi bật nhất trong run này là learning rate: đổi duy nhất từ 1e-4 xuống 1e-5 khiến target và format về 0. Placement q,v với rank match ngân sách lại gần như hòa/thắng `text-linear` trên task hẹp, nên cần thận trọng khi suy rộng kết luận đó. Cuối cùng, QLoRA tiết kiệm bộ nhớ rõ rệt nhưng đổi bằng target và latency, vì vậy không đáng dùng khi T4 đã chứa được LoRA 16-bit.

Ba điều tôi học được:

1. Mask là điều kiện nhân quả đầu tiên: một loss curve đẹp không có ý nghĩa nếu prompt đang bị supervise.
2. Baseline prompt mạnh phải được đo trước train; kết quả FT +0.200 chỉ có giá trị vì (b) đã được đóng băng ở 0.765.
3. Target score cao không tự động là deployable: regression gate phát hiện rõ sự chuyên biệt hóa quá mức của adapter này.

Nếu có thêm hai giờ, tôi sẽ thêm 1–5% replay dữ liệu phổ thông, chạy lại `correct` với cùng budget và kiểm tra liệu regression có trở lại trong tolerance mà không làm mất cải thiện target hay không.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
