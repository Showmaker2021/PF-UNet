# PF-UNet: Học luật điều phối như một trường ưu tiên bằng U-Net cho lập lịch xưởng linh hoạt

*Bản nháp đề cương nghiên cứu, 03/11/2026. Các nhận định về novelty dựa trên một lượt tra cứu giới hạn, cần kiểm chứng lại trước khi cam kết.*

## 1. Tóm tắt

Các bộ giải nơ-ron cho Flexible Job Shop Scheduling (FJSP) hiện nay, từ HGNN, DANIEL đến ReSched (ICLR 2026), đều biểu diễn trạng thái bằng đồ thị hoặc chuỗi token thực thể rồi dùng attention hoặc GNN. Chi phí mỗi bước tăng theo số operation, và chất lượng khi huấn luyện trên instance nhỏ rồi chạy trên instance lớn còn suy giảm. Đề cương này đề xuất **PF-UNet (Priority-Field U-Net)**: rasterize các ứng viên điều phối lên một **canvas chuẩn hóa theo từng máy** (trục x là thời điểm bắt đầu hiệu dụng, trục y là mức độ găng của job), chạy một U-Net nhiều mức trên canvas, rồi đọc ngược điểm ưu tiên cho từng cặp (operation, máy) bằng lấy mẫu song tuyến tính. Luận điểm trung tâm: biểu diễn mật độ chuẩn hóa làm cho chi phí mỗi quyết định gần như không phụ thuộc kích thước instance và giúp chuyển từ nhỏ sang lớn tốt hơn. Đây là hướng **novelty vừa đến cao, rủi ro kỹ thuật cao**.

## 2. Giới thiệu và động lực

**Bối cảnh.** ReSched thu gọn trạng thái còn 4 đặc trưng, dùng Transformer hai nhánh (self-attention cho operation với RoPE, cross-attention cho máy), huấn luyện bằng REINFORCE hoặc PPO với reward là mức tăng của cận dưới makespan, và vượt DANIEL trên nhiều benchmark. Tác giả tự nêu hạn chế: chưa có phản hồi máy sang operation, và chưa kết hợp với phương pháp tìm kiếm.

**Vấn đề còn mở (theo các bài mình đã đọc).**

- Chi phí: mỗi bước chạy lại bộ mã hóa trên đồ thị còn lại, tổng khoảng O(n³) attention theo Phụ lục C của ReSched.
- Quy mô lớn: ReSched kiểm tra FJSP tới 40×10; các bài khác cho thấy gap tăng ở 200×5 (ví dụ DANIEL, PLAN với sampling). Tỉ lệ operation:máy tăng làm loãng thông tin ở nhánh máy.
- Các hướng "làm nhanh" (PLAN, CoRS) và "một lượt" (SchedDiff, GOAL) đang đông, nên cần một góc khác.

**Ý tưởng.** Mọi luật điều phối cổ điển (SPT, MWKR, FIFO) là hàm của vài tọa độ: thời điểm sẵn sàng, công việc còn lại, thời gian xử lý. Ta đặt các ứng viên lên mặt phẳng của đúng những tọa độ đó và để một U-Net học **bề mặt ưu tiên** có điều kiện theo mật độ cục bộ và toàn cục.

## 3. Định vị và chấm điểm novelty

### 3.1 So với công trình liên quan

| Công trình | Biểu diễn | Cơ chế chính | Khác biệt của PF-UNet |
| --- | --- | --- | --- |
| HGNN, DANIEL | Đồ thị dị thể, hơn 20 đặc trưng | GNN/attention, PPO | Không dùng đồ thị hay đặc trưng thủ công dày |
| ReSched (ICLR 2026) | Token operation+máy, 4 đặc trưng | Transformer + RoPE + self-based cross-attn | Trường rasterize, U-Net, bỏ token operation |
| ReLA (2026) | Token thực thể | Self-attn + conv + cross-attn đa tỉ lệ | Phân giải theo mặt phẳng trường, không theo token |
| PLAN (2026) | Đặc trưng kiểu DANIEL | Liquid-inspired + MHA nhẹ | Không nhắm hiệu năng tham số |
| CoRS (2026) | Gần ReSched | Tái dùng trạng thái | Cùng mục tiêu giảm chi phí, khác cơ chế (cần đọc kỹ) |
| SchedDiff (NeurIPS 2026) | Priority một lượt | Diffusion + active list scheduling | PF-UNet là chính sách từng bước, huấn luyện RL |
| GOAL (2026) | Đồ thị quan hệ | Diffusion có điều kiện | Khác paradigm |
| Graph U-Net (2019) | Đồ thị bất kỳ | gPool/gUnpool | Dùng lưới thật thay vì pooling trên đồ thị |
| Scale-Net (AAAI 2026, VRP đa nhiệm) | Token nút, đồ thị đầy đủ | Graph U-Net: top-k pooling, unpool, skip, nối đặc trưng mọi mức vào decoder | Lưới rasterize chuẩn hóa thay vì pooling theo số nút tuyệt đối; bài toán là lập lịch |

Tiền lệ cần lưu ý: biểu diễn trạng thái dạng ảnh cho lập lịch (Deep RM cho cluster scheduling) và chuỗi rasterize-rồi-CNN trong thị giác 3D (PointPillars), cùng RoIAlign để đọc đặc trưng tại tọa độ thực. Mình chưa tìm thấy việc rasterize **mật độ ứng viên trên mặt phẳng (thời điểm bắt đầu hiệu dụng, độ găng) theo từng máy** cho FJSP, nhưng khảo sát chưa đầy đủ.

### 3.2 Chấm điểm tự đánh giá (thang 1-5, chủ quan)

| Tiêu chí | Điểm | Lý do |
| --- | --- | --- |
| Novelty biểu diễn | 4 | Trường mật độ theo máy chưa thấy trong tra cứu; có tiền lệ gần ở các lĩnh vực khác |
| Novelty kiến trúc | 2-3 | U-Net, RoIAlign, attention giữa máy đều đã biết; ghép lại cho lập lịch mới; Scale-Net đã dùng U-Net cho VRP nên riêng việc dùng U-Net không còn mới |
| Novelty bài toán/MDP | 1-2 | Cố ý dùng lại MDP và reward của ReSched để so sánh công bằng |
| Cơ sở lý thuyết | 2 | Lập luận hội tụ mật độ mới ở mức trực giác, cần làm chặt hoặc hạ xuống thành quan sát thực nghiệm |
| Tiềm năng đóng góp thực nghiệm | 4 nếu H1 và H2 đúng | Chi phí phẳng theo n và gap phẳng theo n là kết quả rõ ràng |
| Rủi ro kỹ thuật | 4 (cao) | Rasterize mất chi tiết, tóm tắt phần đuôi có thể mất thông tin |

**Đánh giá tổng quát:** khoảng 3.3/5 (hạ nhẹ sau khi đối chiếu Scale-Net) về novelty với phương sai lớn. Nếu pilot thất bại thì xuống mức trung bình; nếu H1-H2 đúng thì đủ mạnh cho hội nghị tốt hoặc tạp chí.

### 3.3 Bài học từ Scale-Net (AAAI 2026)

Scale-Net (Liu, Cao, Yin, Ong) đưa Graph U-Net vào bộ giải VRP đa nhiệm: encoder chạy attention trên các tập nút co dần nhờ top-k pooling, unpool bằng zero-fill kèm skip connection, và decoder nối đặc trưng của mọi mức vào ngữ cảnh. Bài này tác động đến đề cương theo bốn hướng.

**1. Novelty giảm ở phần kiến trúc.** Việc dùng U-Net cho NCO không còn mới. Điểm mới của PF-UNet phải nằm ở biểu diễn trường và ở bài toán lập lịch. Phương án dự phòng "Graph U-Net theo job" giờ gần Scale-Net, nên novelty thấp hơn mình ước lượng trước đây; chỉ dùng làm phương án cuối.

**2. Bằng chứng ủng hộ hướng đa mức.** Scale-Net cải thiện POMO-MTL ở 16/16 tác vụ quy mô nhỏ nhưng mức tăng bé (CVRP-100: gap 1.846% xuống 1.723%; VRPTW-100: 5.313% xuống 5.067%). Ở quy mô lớn sau khi fine-tune, mức tăng rõ hơn (Set-X 500-1000 nút: 9.85% so với 13.49% của POMO-MTL-1k). Họ cũng ước tính bộ nhớ attention của encoder giảm khoảng 65% ở 5000 nút (4.6 GB xuống 1.6 GB), đây là tính toán giải tích chứ không phải số đo thực.

**3. Bằng chứng cảnh báo cho "học nhỏ, test lớn".**

- Zero-shot từ 100 nút lên Set-X 500-1000 nút, Scale-Net (23.45%) kém POMO-MTL (16.77%); chỉ sau khi fine-tune ở 1000 nút mới vượt.
- Bảng 9: mô hình chỉ huấn luyện ở 100 nút, dù đã chọn lịch pooling riêng cho từng cỡ, gap tăng từ 13.44% (khoảng 500 nút) lên 61.43% (khoảng 1000 nút).
- Kích thước pooling là số nút tuyệt đối ({75, 50, 25} khi huấn luyện, {500, 250, 125} khi fine-tune), tức không bất biến theo quy mô.
- Mô hình fine-tune ở 3000 nút kém đi khi quay lại 1000 nút (quên), và Scale-Net-1k sụp ở 5000 nút (CVRP: 159.2 so với 80.0 của POMO-MTL-1k).

Hệ quả: U-Net với pooling tuyệt đối không tự cho khả năng chuyển quy mô. PF-UNet dùng lưới chuẩn hóa nên về nguyên tắc tránh lỗi này, nhưng số ứng viên trên mỗi pixel tăng theo n, một dạng lệch phân phối tương tự, nên phải chuẩn hóa mật độ (mục 5.2).

**4. Ý có thể mượn.**

- Nối đặc trưng của mọi mức decoder vào đầu quyết định (mục 5.4).
- Giao thức hai giai đoạn: huấn luyện nhỏ, rồi fine-tune dần lên quy mô lớn với tốc độ học giảm, và đo hiện tượng quên (mục 6.1, E9).
- Dựng một baseline kiểu Scale-Net (top-k pooling trên token) để tách "do U-Net" khỏi "do biểu diễn trường" (E6b).
- Phân tích bộ nhớ có công thức và kiểm định ý nghĩa thống kê theo cặp (E11).

## 4. Chi tiết ý tưởng

### 4.1 Giả thuyết cần kiểm chứng

- **H1:** PF-UNet huấn luyện trên 10×5 đạt chất lượng ngang ReSched ở 30×10 và 40×10, tốt hơn ở 100×10 và 200×5.
- **H2:** thời gian mỗi quyết định gần như phẳng theo n (do chi phí U-Net không phụ thuộc n, chỉ rasterize phụ thuộc số job).
- **H3:** gap theo n phẳng hơn các mô hình attention khi n tăng.
- **H4:** bề mặt ưu tiên học được có cấu trúc diễn giải được (gần SPT ở vùng thưa, gần MWKR ở vùng găng).

### 4.2 Điều không đổi so với ReSched (có chủ đích)

MDP dựng lời giải từng bước, hành động là cặp (operation, máy), reward bằng âm mức tăng cận dưới makespan, dữ liệu SD1/SD2, giao thức đánh giá greedy và sampling 100 quỹ đạo. Thay đổi nằm ở **trạng thái quan sát và mạng**.

## 5. Kiến trúc chi tiết

### 5.0 Tổng quan pipeline

PF-UNet đổi mỗi trạng thái xưởng thành một bộ canvas chuẩn hóa theo từng máy, chạy U-Net một lượt, rồi đọc ngược điểm ưu tiên cho từng cặp (operation, máy).

&#91;embedded content: pipeline PF-UNet · 7 tầng, 1 vòng lặp\]

Đọc từ trái sang phải, từ trên xuống: ba tầng đầu biến trạng thái thành ảnh (cố định, không có tham số học), vùng nền xám là mạng U-Net (học), ba tầng cuối đọc điểm, thực hiện hành động rồi quay vòng sang bước kế tiếp.

**Phần cố định:** tọa độ chuẩn hóa, rasterize, lấy mẫu ở readout. **Phần học được:** U-Net, attention giữa các máy, MLP của đầu chính sách và critic.

**Vì sao thiết kế như vậy.**

- Hai trục (thời điểm bắt đầu hiệu dụng, công việc còn lại) trùng với biến của các luật điều phối cổ điển, nên trường học được là một dạng tổng quát hóa của chúng.
- U-Net có hai vai trò: mức thô nắm tắc nghẽn toàn cục, mức mịn xử lý xung đột cục bộ; skip connection giữ chi tiết.
- Readout cộng đặc trưng thô của cặp để phân biệt các operation rơi vào cùng một pixel.

### 5.1 Ứng viên và đặc trưng

Ở bước t, ứng viên là operation đứng đầu của mỗi job còn việc. Với job j gọi R\_j là thời điểm sẵn sàng, A\_k là thời điểm máy k rảnh, p\_ok là thời gian xử lý, W\_j = tổng thời gian xử lý tối thiểu của mọi operation còn lại của j (gồm cả operation đầu).

- t0 = min\_k A\_k (trừ mốc như relative available time của ReSched).
- ST\_ok = max(R\_j, A\_k) là thời điểm bắt đầu hiệu dụng của (o, k).
- LB\_t = max( max\_j (max(R\_j, t0) + W\_j), max\_k A\_k ) là cận dưới makespan còn lại, S\_t = max(LB\_t − t0, ε) là hệ số chuẩn hóa.
- **Tọa độ trên canvas máy k:** x\_ok = clip((ST\_ok − t0)/S\_t, 0, 1), y\_o = clip(W\_j/S\_t, 0, 1). Máy rảnh muộn làm ứng viên dịch sang phải, nên tình trạng máy nằm trong hình học.

### 5.2 Rasterize theo máy

Mỗi máy k có một canvas H×W (mặc định 64×64, pilot 32×32), gồm khoảng 8 kênh:

1. mật độ ứng viên (splat song tuyến tính, cộng dồn),
2. tổng p\_ok/S\_t,
3. tổng độ linh hoạt |M\_o|/M,
4. tổng thời gian chờ (ST\_ok − A\_k)/S\_t,
5. số operation còn lại của job (chuẩn hóa),
6. A\_k − t0 chuẩn hóa (kênh hằng),
7. tải tương lai kỳ vọng của máy: Σ p\_o'k/|M\_o'| trên các operation chưa đến lượt tương thích với k, chia tổng tải của mọi máy,
8. mật độ toàn cục trung bình qua các máy (ngữ cảnh).

Hàm rasterize hoán vị bất biến theo job, và trọng số dùng chung giữa các máy nên hoán vị bất biến theo máy.

**Chuẩn hóa mật độ theo quy mô (rút ra từ Scale-Net).** Số ứng viên trên mỗi pixel tăng theo số job J, tạo lệch phân phối đầu vào khi chuyển từ instance nhỏ sang lớn, giống cách Scale-Net bị lệch khi kích thước pooling tuyệt đối đổi. Vì vậy chia kênh mật độ cho J (thành khối lượng xác suất), chia các kênh tổng cho số ứng viên, và thêm một kênh hằng log(J)/log(J\_max) để mạng biết quy mô. Cần ablation có và không có chuẩn hóa này.

### 5.3 U-Net

- 4 mức, kênh \[32, 64, 128, 256\], mỗi mức 2×(Conv3×3 + GroupNorm + GELU), giảm độ phân giải bằng conv stride 2, tăng bằng upsample song tuyến tính + conv, skip nối kênh.
- **Bottleneck liên máy:** pool trung bình bottleneck mỗi máy thành vector g\_k, chạy 2 lớp self-attention giữa M vector, rồi FiLM ngược vào bản đồ bottleneck. Đây là luồng thông tin liên máy và top-down mà ReSched thiếu.
- Critic (cho PPO): pool bottleneck toàn bộ, MLP ra giá trị.

### 5.4 Đọc ngược và đầu quyết định

Với mỗi cặp khả thi (o, k): lấy mẫu song tuyến tính đặc trưng decoder tại (x\_ok, y\_o) ở hai mức (độ phân giải đầy đủ và 1/2), nối với đặc trưng thô của cặp \[p\_ok/S\_t, p\_ok/p\_o^min, (ST\_ok − A\_k)/S\_t, x, y, |M\_o|/M, n\_j/N\_j0\] và g\_k, qua MLP 3 lớp ra điểm. Che mặt nạ cặp không khả thi, softmax trên toàn bộ cặp. Đặc trưng thô giúp phân biệt các operation rơi vào cùng một pixel.

**Biến thể từ Scale-Net.** Nối thêm vector tóm tắt (pool trung bình) của từng mức decoder vào đầu quyết định, giống cách Scale-Net nối đặc trưng của mọi mức vào ngữ cảnh. Đưa vào ablation: chỉ lấy mẫu tại một mức, tại hai mức, hoặc thêm tóm tắt của mọi mức.

### 5.5 Độ phức tạp (ước lượng thô, cần đo)

Mỗi bước: rasterize và đọc ngược O(Σ\_o |M\_o|) trên J ứng viên; U-Net O(M·H·W·C²) không phụ thuộc số operation. Tổng cho n bước: O(n·(J·m̄ + M·H·W·C²)). Phần mở rộng sau: gộp các cặp hoán đổi được (job khác và máy khác) trong một lần forward để giảm số bước.

### 5.6 Mã giả cốt lõi

```python
def rasterize(xy, feats, valid, H, W):
    # xy:[B,M,N,2] trong [0,1]; feats:[B,M,N,C]; valid:[B,M,N]
    gx, gy = xy[...,0]*(W-1), xy[...,1]*(H-1)
    x0, y0 = gx.floor().long(), gy.floor().long()
    wx, wy = gx - x0, gy - y0
    canvas = feats.new_zeros(B*M, C, H*W)
    for dx, dy, w in [(0,0,(1-wx)*(1-wy)), (1,0,wx*(1-wy)),
                      (0,1,(1-wx)*wy),     (1,1,wx*wy)]:
        idx = (y0+dy).clamp(0,H-1)*W + (x0+dx).clamp(0,W-1)
        canvas.scatter_add_(2, idx_expand(idx), feats_flat*(w*valid)[...,None])
    return canvas.view(B*M, C, H, W)

def readout(dec_feat, xy):
    grid = torch.stack([xy[...,0]*2-1, xy[...,1]*2-1], -1)  # [B*M,N,1,2]
    return F.grid_sample(dec_feat, grid, mode='bilinear', align_corners=True)
```

### 5.7 Kích thước tensor qua từng tầng

Ví dụ với M=5 máy, J=10 job, canvas 64×64, 8 kênh đầu vào; B là số instance trong batch.

| Tầng | Đầu vào | Phép toán | Đầu ra |
| --- | --- | --- | --- |
| Ứng viên | R\_j, A\_k, p\_ok, mặt nạ tương thích | Chọn operation đầu của mỗi job, tính ST\_ok, S\_t và tọa độ | Tọa độ \[B,M,J,2\]; đặc trưng splat \[B,M,J,5\]; mặt nạ \[B,M,J\] |
| Rasterize | Tọa độ, đặc trưng | Splat song tuyến tính (scatter\_add) | Canvas \[B·M, 8, 64, 64\] |
| Encoder | Canvas | 4 mức conv, stride 2 giữa các mức | Skip \[B·M,32,64,64\], \[B·M,64,32,32\], \[B·M,128,16,16\]; bottleneck \[B·M,256,8,8\] |
| Bottleneck liên máy | Bản đồ bottleneck | Pool theo máy, self-attention 2 lớp giữa M máy, FiLM ngược | g\_k \[B,M,256\]; bản đồ \[B·M,256,8,8\] |
| Decoder | Bottleneck và skip | Upsample, nối skip, conv từng mức | \[B·M,32,64,64\] và \[B·M,64,32,32\] |
| Readout | Bản đồ decoder, tọa độ | Lấy mẫu song tuyến tính ở 2 mức, nối 7 đặc trưng thô và g\_k | \[B,M,J,359\] (32+64+7+256) |
| Đầu chính sách | Vector readout | MLP 3 lớp, che cặp không khả thi, softmax | Điểm \[B,M,J\]; phân phối trên các cặp khả thi |
| Critic (PPO) | g\_k hoặc pool toàn bộ bottleneck | MLP | Giá trị V \[B\] |

Đổi J=10 thành J=200 chỉ làm đổi chiều J của tensor ứng viên và readout; kích thước các tensor trong U-Net giữ nguyên.

## 6. Chi tiết triển khai

### 6.1 Huấn luyện

- PPO (bản PPO của ReSched đạt kết quả tốt hơn REINFORCE trong Phụ lục E), siêu tham số khởi đầu theo DANIEL: γ=1, λ=0.98, clip 0.2, hệ số value 0.5, entropy 0.01. Reward là âm mức tăng cận dưới makespan.
- Dữ liệu: SD1 (D\~U\[1,20\], số operation mỗi job \~U\[0.8n,1.2n\]) và SD2 (D\~U\[1,99\], số operation \~U\[1,n\]) theo Phụ lục B.2 của ReSched; huấn luyện chủ yếu 10×5, ngân sách nhỏ khoảng 80 nghìn quỹ đạo hiệu dụng và lớn khoảng 2 triệu.
- Tăng cường: nhân thời gian xử lý với hệ số ngẫu nhiên (kiểm tra bất biến tỉ lệ), hoán vị job/máy.

**Giao thức hai giai đoạn (theo Scale-Net).** Ngoài chế độ zero-shot (chỉ huấn luyện 10×5), thêm chế độ fine-tune tuần tự lên 20×10, 40×10, 100×10 với tốc độ học giảm dần (Scale-Net dùng 1e-5, 1e-6, 5e-7 cho 1000, 2000, 3000 nút), và luôn đánh giá lại ở quy mô nhỏ để đo hiện tượng quên. Nếu cần chống quên, trộn lại một phần instance nhỏ khi fine-tune.

### 6.2 Kiểm thử đơn vị bắt buộc

1. Hoán vị job và máy cho điểm quyết định không đổi.
2. Nhân mọi thời gian xử lý với hằng c không đổi quyết định greedy.
3. Vẽ canvas cho vài instance, đảm bảo điểm không dồn vào vài pixel.
4. Overfit một instance trong vài trăm bước.
5. So tổng makespan môi trường tự dựng với môi trường gốc của ReSched trên cùng chuỗi hành động.

### 6.3 Làm việc trên code ReSched

**Có, nên fork ngay từ repo ReSched** (URL theo bài báo: github.com/XiangjieXiao/ReSched; mình chưa mở được repo nên chưa kiểm tra cấu trúc và giấy phép). Kế hoạch:

1. Tái hiện Bảng 1 (SD1 10×5) bằng code gốc để chắc pipeline đúng, đặt git tag "baseline".
2. Giữ nguyên môi trường, sinh dữ liệu, reward, đánh giá greedy/sampling, ghi log.
3. Thêm module `observation_builder` (đọc từ môi trường: R\_j, A\_k, ma trận thời gian xử lý, mặt nạ tương thích) và module `pfunet.py`; thay mô hình, giữ nguyên giao diện hành động.
4. Nếu môi trường gốc dựng tensor đồ thị khó đọc ngược, phương án phụ là dựa trên môi trường của DANIEL (đã xác minh có tệp `fjsp_env_same_op_nums.py`, `fjsp_env_various_op_nums.py`, `data_utils.py`, `ortools_solver.py`), nhưng lưu ý DANIEL có cắt tỉa theo thời gian hiện tại.
5. Kiểm tra giấy phép và ghi trích dẫn đầy đủ.

### 6.4 Lộ trình 8 tuần

| Tuần | Việc |
| --- | --- |
| 1 | Tái hiện ReSched; viết bộ rasterizer; vẽ canvas |
| 2 | U-Net + readout; test đơn vị; overfit một instance |
| 3 | Huấn luyện 10×5 SD1/SD2; so với ReSched (pilot, ngưỡng dừng) |
| 4 | OOD 30×10, 40×10, 100×10, 200×5 |
| 5 | Benchmark công khai Brandimarte, Hurink; JSSP Taillard/DMU |
| 6 | Ablation |
| 7 | Diễn giải bề mặt ưu tiên, đường cong tỉ lệ |
| 8 | Viết bài, chuẩn bị mã nguồn |

### 6.5 Đặc tả chính xác cho coding agent

Mục 6.5 đến 6.9 là hợp đồng giữa đề cương và code. Nếu mô tả ở các mục trước mâu thuẫn với các mục này, làm theo 6.5 đến 6.9. Công thức chuẩn hóa ở 5.1 đã được sửa cho khớp: định nghĩa cũ dùng R\_j thay cho max(R\_j, t0) nên y có thể vượt 1.

**Biến trạng thái ở bước t**

- Job j còn việc có operation đầu (head) chưa xếp là o\_j. R\_j là thời điểm hoàn thành của operation xếp gần nhất của job (0 nếu chưa xếp operation nào).
- Máy k có A\_k là thời điểm rảnh (0 lúc đầu), bằng thời điểm hoàn thành của operation cuối cùng xếp lên máy k. Không chèn vào khoảng trống, đúng như Phương trình (2) của ReSched.
- p\_ok > 0 chỉ khi máy k tương thích với operation o; ngược lại cặp (o, k) không tồn tại (mặt nạ).
- W\_j là tổng min\_k p trên mọi operation chưa xếp của job j (gồm cả head). n\_j là số operation chưa xếp, N\_j0 là số operation ban đầu của job.
- L\_k = tổng p\_o'k / |M\_o'| trên mọi operation chưa xếp của mọi job, trừ các head, mà tương thích với máy k.
- J\_a là số job còn việc, M là số máy thật của instance.

**Chuẩn hóa (định nghĩa chính thức)**

- t0 = min\_k A\_k, R̃\_j = max(R\_j, t0).
- LB\_t = max( max\_j (R̃\_j + W\_j), max\_k A\_k ), S\_t = max(LB\_t − t0, 1e-6).
- ST\_ok = max(R\_j, A\_k), x\_ok = (ST\_ok − t0) / S\_t, y\_j = W\_j / S\_t. Cả hai nằm trong \[0, 1\]: vì LB\_t ≥ R̃\_j + W\_j nên W\_j ≤ S\_t và R\_j ≤ LB\_t, còn A\_k ≤ LB\_t theo định nghĩa.

**Hành động và chuyển trạng thái.** Chọn cặp hợp lệ (o\_j, k). start = ST\_ok, finish = start + p\_ok, rồi R\_j ← finish, A\_k ← finish. Operation o\_j bị loại, operation kế tiếp của job thành head; job hết operation thì bị loại khỏi tập ứng viên. Kết thúc khi mọi job xong, makespan là finish lớn nhất.

**Reward.** Dùng nguyên hàm reward của ReSched (âm mức tăng của cận dưới makespan ước lượng, Phương trình 3). Không tự viết lại. Nếu cần đối chiếu, cận dưới này có vẻ tương đương max\_j (R\_j + W\_j) trên mọi job kể cả job đã xong (W\_j = 0); cần xác nhận trong code gốc.

**Tám kênh canvas của máy k.** Kênh splat được cộng dồn theo trọng số song tuyến tính của tọa độ (x\_ok, y\_j), rồi chia cho J\_a.

| Kênh | Nội dung | Cách tạo |
| --- | --- | --- |
| 1 | Mật độ: giá trị 1 cho mỗi ứng viên (o, k) | Splat, chia J\_a |
| 2 | p\_ok / S\_t | Splat, chia J\_a |
| 3 | Độ linh hoạt | M\_o |
| 4 | Thời gian chờ (ST\_ok − A\_k) / S\_t | Splat, chia J\_a |
| 5 | Tiến độ n\_j / N\_j0 | Splat, chia J\_a |
| 6 | Máy rảnh (A\_k − t0) / S\_t | Hằng trên toàn canvas |
| 7 | Tỉ lệ tải tương lai L\_k / (tổng L\_k' qua các máy + 1e-6) | Hằng trên toàn canvas |
| 8 | Ngữ cảnh: trung bình qua các máy thật của kênh 1 sau chuẩn hóa | Một bản đồ chung, sao chép cho mọi máy |

Kênh tùy chọn thứ 9 dùng cho ablation chuẩn hóa quy mô: giá trị hằng log(J\_a) / log(J\_max), với J\_max mặc định 256. Khi bật thì canvas có 9 kênh và `C_in` trong cấu hình phải đổi theo.

**Bảy đặc trưng thô của cặp (cho đầu quyết định).** p\_ok / S\_t, p\_ok / p\_o^min, (ST\_ok − A\_k) / S\_t, x\_ok, y\_j, |M\_o| / M, n\_j / N\_j0.

**Trường hợp biên**

- Job hết operation không phải ứng viên và không được splat. Khi mọi job xong thì dừng, không gọi mạng.
- Máy không tương thích với ứng viên nào vẫn đi qua U-Net với canvas chỉ có các kênh hằng; không có cặp nên không ảnh hưởng đầu ra.
- Padding theo batch: dùng M\_max và J\_max của batch, kèm mặt nạ cho máy giả và ứng viên giả. Máy giả bị loại khỏi attention ở bottleneck, khỏi kênh 7 và kênh 8; ứng viên giả không được splat và không tạo cặp.
- Chỉ số bilinear ngoài biên thì kẹp (clamp) vào \[0, H−1\] hoặc \[0, W−1\]; trọng số vẫn giữ nguyên nên tổng khối lượng của mỗi ứng viên bằng 1.
- Khi greedy gặp hòa điểm, chọn cặp có chỉ số phẳng nhỏ nhất.
- Dùng float32; thời gian xử lý lưu dạng số thực nhưng giá trị là số nguyên.
- Mỗi lần chạy ghi seed riêng cho numpy, torch và sinh dữ liệu.

### 6.6 Cấu trúc repo và giao diện module

Tên thư mục và hàm dưới đây là đề xuất; nếu repo ReSched có quy ước khác thì giữ quy ước của repo nhưng giữ nguyên tên và ý nghĩa của các giao diện.

```text
pfunet/
  configs/        # yaml: model, train, eval
  envs/           # bọc môi trường ReSched, KHÔNG sửa logic gốc
    obs_builder.py    # build_obs(env_state) -> Obs
  models/
    rasterize.py      # rasterize(), readout_sample()
    unet.py           # UNet2D: encoder, decoder, skip, FiLM
    pfunet.py         # PFUNetPolicy
  train/
    ppo.py            # vòng lặp PPO
    eval.py           # greedy, sampling 100, đo thời gian
  scripts/        # reproduce_resched.sh, run_pilot.sh
  tests/          # test_env_parity, test_rasterize, test_invariance
```

**Kiểu dữ liệu `Obs`** (một batch B instance, đã padding theo M\_max và J\_max; ký hiệu B, M, J dưới đây là các kích thước đã padding)

| Trường | Hình dạng | Ý nghĩa |
| --- | --- | --- |
| `xy` | \[B, M, J, 2\] | Tọa độ (x\_ok, y\_j) trong \[0, 1\] |
| `feats` | \[B, M, J, 5\] | Giá trị splat của kênh 1 đến 5 |
| `raw` | \[B, M, J, 7\] | Bảy đặc trưng thô của cặp |
| `mask_pair` | \[B, M, J\] bool | True nếu cặp (job j, máy k) khả thi |
| `mask_machine` | \[B, M\] bool | True nếu máy là máy thật |
| `mask_job` | \[B, J\] bool | True nếu job còn việc |
| `const` | \[B, M, 2\] | Kênh 6 và kênh 7 của từng máy |
| `scalars` | \[B, 2\] | J\_a và S\_t (để chuẩn hóa và gỡ lỗi) |

**Chữ ký hàm**

- `build_obs(state) -> Obs`: vector hóa hoàn toàn trên batch, không vòng lặp Python theo instance. Có bản tham chiếu viết bằng Python thuần để kiểm thử tương đương.
- `rasterize(obs, H, W) -> Tensor[B*M, C_in, H, W]`: splat song tuyến tính bằng `scatter_add_` hoặc `index_put_(accumulate=True)`, chia J\_a, ghép kênh hằng và kênh ngữ cảnh.
- `readout_sample(dec_maps, xy) -> Tensor[B, M, J, C]`: dùng `grid_sample` với `align_corners=True`, ánh xạ x thành 2x − 1 và y thành 2y − 1, khớp với cách splat dùng x·(W−1) và y·(H−1). Lấy mẫu từ hai mức decoder (độ phân giải đầy đủ và một nửa) rồi nối kênh.
- `PFUNetPolicy.forward(obs) -> (logits[B, M, J], value[B])`: logits bằng −inf ở các cặp không khả thi.
- `PFUNetPolicy.evaluate(obs, action) -> (logp, entropy, value)`: dùng lại trong các epoch PPO.

**Quy ước chỉ số hành động.** Chỉ số phẳng a = k·J + j, trong đó k là máy và j là job. Môi trường nhận cặp (j, k) và tự suy ra operation head của job j.

**Lưu ý hiệu năng.** Canvas và bản đồ U-Net không được lưu vào bộ đệm huấn luyện; chỉ lưu `Obs` (nhỏ) rồi tính lại khi chạy các epoch PPO.

### 6.7 Cấu hình mặc định

Các giá trị dưới đây là điểm xuất phát. Khi ghi chú ghi "tự chọn", đó là lựa chọn của đề cương, không lấy từ ReSched; agent phải ghi lại và có thể quét.

| Khóa | Mặc định | Ghi chú |
| --- | --- | --- |
| `canvas.H`, `canvas.W` | 64 (pilot: 32) | Độ phân giải canvas |
| `canvas.C_in` | 8 | 9 nếu bật kênh log(J\_a) |
| `unet.channels` | \[32, 64, 128, 256\] | 4 mức, 3 lần giảm độ phân giải |
| `unet.blocks_per_level` | 2 | Conv3×3, GroupNorm, GELU |
| `unet.groupnorm_groups` | 8 | Tự chọn |
| `bottleneck.attn_layers` | 2 | Self-attention giữa các máy |
| `bottleneck.attn_heads` | 4 | Tự chọn |
| `bottleneck.dim` | 256 | Bằng số kênh ở mức sâu nhất |
| `readout.levels` | \[0, 1\] | Decoder ở độ phân giải đầy đủ và một nửa |
| `head.hidden` | \[128, 64\] | MLP 3 lớp, đầu ra 1 điểm cho mỗi cặp |
| `J_max_log` | 256 | Chỉ dùng cho kênh 9 |
| `ppo.gamma` | 1.0 | Theo DANIEL (Phụ lục E.2 của ReSched) |
| `ppo.lambda` | 0.98 | Theo DANIEL |
| `ppo.clip` | 0.2 | Theo DANIEL |
| `ppo.vf_coef`, `ppo.ent_coef` | 0.5 và 0.01 | Theo DANIEL |
| `ppo.epochs` (K) | 4 | Theo Phụ lục E |
| `ppo.batch` (B) | 50 instance mỗi update | Theo Phụ lục E |
| `ppo.updates` (U) | 400 (ngân sách nhỏ), 10000 (ngân sách lớn) | 400×50×4 = 80 nghìn, 10000×50×4 = 2 triệu quỹ đạo hiệu dụng |
| `optim.lr` | 5e-5 | ReSched dùng giá trị này cho REINFORCE; quét {2e-5, 5e-5, 1e-4} |
| `optim.grad_clip` | 1.0 | Tự chọn |
| `ppo.minibatch` | 10 instance | Tự chọn |
| `eval.every` | 10 update | Đánh giá trên tập validation |
| `val.size` | 100 instance | Seed sinh dữ liệu khác tập huấn luyện và tập test |
| `eval.decode` | greedy và sampling 100 | Như ReSched |

### 6.8 Vòng lặp huấn luyện PPO

```python
for update in range(U):
    env.reset(sample_instances(B))            # B instance cùng cỡ
    buf = []
    while not env.all_done():
        obs = build_obs(env.state)
        with torch.no_grad():
            logits, value = policy(obs)
        dist = Categorical(logits=logits.flatten(1))   # chỉ số phẳng k*J + j
        a = dist.sample()
        reward, done = env.step(a)             # env suy (j, k) từ a
        buf.append(dict(obs=obs, a=a, logp=dist.log_prob(a),
                        v=value, r=reward, done=done))
    adv, ret = gae(buf, gamma, lam)            # chuẩn hóa adv theo batch
    for _ in range(K):
        for mb in minibatches(buf, adv, ret):
            logp, ent, v = policy.evaluate(mb.obs, mb.a)
            ratio = (logp - mb.logp).exp()
            pg = -torch.min(ratio * mb.adv,
                            ratio.clamp(1 - eps, 1 + eps) * mb.adv).mean()
            vl = (v - mb.ret).pow(2).mean()
            loss = pg + vf_coef * vl - ent_coef * ent.mean()
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(policy.parameters(), grad_clip); opt.step()
    if update % eval_every == 0:
        evaluate_on_validation(); save_if_best()
```

Đây là mã giả, chưa được chạy thử. Các điểm cần xử lý khi cài đặt:

- Với SD2, số operation mỗi job thay đổi nên các instance trong cùng batch kết thúc ở bước khác nhau. Instance đã xong phải bị che (không bước, không tính vào loss), và `gae` phải dừng đúng chỗ của từng instance.
- Bộ đệm chỉ lưu `Obs`, hành động, log-prob, giá trị, reward và cờ done. Không lưu canvas hay bản đồ U-Net.
- Ví dụ ước lượng bộ nhớ: B = 50, M = 5, J = 200 thì mỗi bước `Obs` có khoảng 700 nghìn số thực (14 giá trị cho mỗi cặp); với khoảng 1000 bước là khoảng 2.8 GB ở float32. Giảm B hoặc lưu `Obs` ở float16 khi chạy cỡ lớn.
- Chi phí một update gồm n lần forward khi thu thập và K × n lần forward cộng backward khi cập nhật; đây là chi phí chính cần đo ở bước 6.9.

### 6.9 Mốc nghiệm thu và quy tắc làm việc cho agent

**Mốc nghiệm thu.** Chỉ sang mốc sau khi mốc trước đạt. Nếu một mốc không đạt, dừng và báo cáo, không chỉnh ngưỡng.

| Mốc | Việc | Tiêu chí đạt |
| --- | --- | --- |
| M0 | Tái hiện ReSched trên SD1 10×5, huấn luyện lại bằng code gốc | Gap greedy và sampling nằm trong ±1.5 điểm phần trăm so với 12.25% và 5.98% ở Bảng 1 (dao động giữa các seed của bài gốc khoảng 1 điểm phần trăm) |
| M1 | Môi trường bọc và `build_obs` | Trên 1000 chuỗi hành động ngẫu nhiên, makespan và mọi R\_j, A\_k trùng khớp tuyệt đối với môi trường gốc; `build_obs` vector hóa khớp bản tham chiếu viết bằng Python thuần |
| M2 | `rasterize` và `readout_sample` | Hoán vị job và máy không đổi đầu ra trong sai số 1e-5; nhân mọi thời gian xử lý với hằng c không đổi quyết định greedy; với kênh 1, tổng trên canvas (trước khi chia J\_a) bằng số ứng viên khả thi của máy đó; tọa độ x, y luôn nằm trong \[0, 1\] |
| M3 | Overfit một instance 10×5 | Sau tối đa 2000 update, makespan của chính sách greedy cách OR-Tools không quá 3% trên chính instance đó |
| M4 | Pilot trên SD1/SD2 10×5, canvas 32×32, 4 seed | Báo gap greedy và sampling; nếu gap trong phân phối kém ReSched hơn 3 điểm phần trăm và ở 100×10 vẫn không thắng thì dừng và chuyển phương án dự phòng (mục 7.4) |
| M5 | Quy mô lớn và ablation | Đường cong gap theo n và thời gian mỗi bước theo n cho PF-UNet, ReSched, DANIEL; ablation theo mục 7.3 (gồm E6b) |

**Điều agent phải làm**

- Không sửa logic môi trường và reward của ReSched. Mọi thay đổi nằm trong thư mục mới; giữ git tag "baseline" cho bản tái hiện.
- Không dùng instance test để chọn siêu tham số; chọn mô hình theo tập validation.
- Với mỗi lần chạy ghi lại commit, seed, cấu hình đầy đủ, phiên bản thư viện, loại GPU và thời gian chạy.
- Chạy lại baseline (ReSched, DANIEL) trên cùng máy và cùng instance cho mọi so sánh.
- Báo cáo cả kết quả xấu và lỗi, không đổi tiêu chí để vượt qua.
- Dùng canvas 32×32 và một GPU cho pilot; chưa tối ưu tốc độ trước khi qua M4.
- Mọi thay đổi thiết kế (kênh, chuẩn hóa, kiến trúc) phải được ghi vào tài liệu này.

**Điều agent phải dừng lại và hỏi**

- Cấu trúc repo ReSched khác giả định, hoặc môi trường không cho đọc R\_j, A\_k và ma trận thời gian xử lý ở dạng tensor.
- Không tái hiện được M0.
- Hàm reward gốc không tương đương mô tả ở 6.5.
- Giấy phép của repo ReSched không cho phép dùng lại hoặc sửa đổi.
- Một mốc không đạt sau hai lần thử hợp lý.

## 7. Thiết kế thực nghiệm

### 7.1 Dữ liệu và baseline

- **Tổng hợp:** SD1, SD2 (huấn luyện 10×5, 20×10; OOD 30×10, 40×10, 100×10, 200×5).
- **Công khai:** Brandimarte, Hurink (edata, rdata, vdata) cho FJSP; Taillard và DMU cho JSSP (huấn luyện 10×10, như ReSched).
- **Baseline:** ReSched, DANIEL, HGNN, các luật SPT/MWKR/MOPNR/FIFO, OR-Tools CP-SAT (giới hạn thời gian như ReSched), nếu có mã thì thêm PLAN và CoRS.

### 7.2 Chỉ số

Gap so với OR-Tools hoặc cận trên tốt nhất đã biết; thời gian suy luận mỗi instance và **mỗi bước theo n**; bộ nhớ; số tham số; chạy ít nhất 4 seed (ReSched báo cáo 4 lần chạy và dao động khoảng 1 điểm phần trăm trên nhiều tập, nên cải thiện dưới ngưỡng đó chưa đáng tin).

### 7.3 Danh sách thí nghiệm

- **E0** Tái hiện ReSched.
- **E1** Trong phân phối SD1 và SD2.
- **E2** Tổng quát hóa theo quy mô tới 100×10, 200×5, và thử 500×10 nếu GPU cho phép.
- **E3** Benchmark công khai FJSP.
- **E4** JSSP: một mô hình, canvas dùng chung (mỗi operation một máy).
- **E5** Đường cong gap(n) và thời gian mỗi bước(n) so với ReSched, DANIEL.
- **E6** Ablation: bỏ skip, giảm số mức, CNN một mức, thay readout bằng pooling toàn cục, độ phân giải 16/32/64, bỏ kênh tải tương lai, bỏ attention liên máy, đổi cách chuẩn hóa S\_t.
- **E7** Diễn giải: vẽ bề mặt ưu tiên và so với SPT/MWKR.
- **E8** Lệch phân phối: huấn luyện SD1, kiểm tra SD2 và ngược lại.

Bổ sung sau khi xem Scale-Net:

- **E6b** Baseline có kiểm soát: thay trường rasterize bằng U-Net kiểu Scale-Net (attention trên token operation, top-k pooling học được, unpool zero-fill, skip) trên cùng MDP và đầu quyết định, để tách "do U-Net" khỏi "do biểu diễn trường".
- **E9** Zero-shot so với fine-tune tuần tự: so đường gap(n) ở hai chế độ và đo hiện tượng quên ở 10×5 sau khi fine-tune ở 100×10.
- **E10** Few-shot sang biến thể mới (ví dụ FFSP, hoặc FJSP với độ linh hoạt máy khác): 10 epoch tinh chỉnh, vẽ gap theo epoch như Scale-Net làm cho VRPBLTW.
- **E11** Phân tích bộ nhớ và FLOPs có công thức kèm số đo thực, và kiểm định ý nghĩa thống kê theo cặp giữa PF-UNet và ReSched.

### 7.4 Tiêu chí thành công và dừng

- **Dừng:** sau pilot 10×5, nếu gap trong phân phối kém ReSched hơn 3 điểm phần trăm và ở 100×10 không thắng, chuyển sang phương án dự phòng.
- **Phương án dự phòng:** Graph U-Net phân cấp với pooling theo job và theo thời gian, unpool kèm phản hồi máy→operation.
- **Thành công mạnh:** H1 và H2 cùng đúng với ít nhất 4 seed.

### 7.5 Giao thức thực nghiệm kế thừa ReSched

Kế thừa giao thức của ReSched để so sánh trực tiếp, chỉ thêm các cỡ lớn mà bài gốc chưa kiểm tra. Số liệu của ReSched dưới đây lấy từ bản ICLR 2026.

**Bộ dữ liệu**

| Bộ | Huấn luyện | Kiểm tra | Nguồn |
| --- | --- | --- | --- |
| FJSP trong phân phối (SD1, SD2) | 10×5, 20×5, 15×10, 20×10 | 100 instance cùng cỡ | Kế thừa |
| FJSP ngoài phân phối (SD1, SD2) | 10×5 và 20×10 | 30×10, 40×10 | Kế thừa |
| FJSP benchmark | SD1 (10×5, 15×10) | Brandimarte, Hurink (edata, rdata, vdata); gap so với upper bound tốt nhất đã biết (Behnke & Geiger 2012) | Kế thừa |
| FJSP cỡ lớn | 10×5 (và 20×10) | 100×10, 200×5, lớn hơn nếu GPU cho phép | Mới, tự chạy lại baseline |
| JSSP | Dữ liệu tổng hợp 10×10 (phân phối của L2D) | Taillard 15×15 tới 100×20; DMU 20×15 tới 50×20; tổng hợp 6×6 tới 100×20 | Kế thừa |
| FFSP | Cỡ 20 (3 stage, 4 máy mỗi stage) | Cỡ 20, 50, 100; greedy và sampling | Kế thừa |

**Ngân sách và giao thức**

| Mục | ReSched | PF-UNet đề xuất |
| --- | --- | --- |
| Thuật toán | REINFORCE (chính), PPO (Phụ lục E) | PPO; thêm REINFORCE nếu đủ thời gian để cô lập ảnh hưởng của thuật toán |
| Ngân sách | REINFORCE: 2000 epoch × 1000 instance, batch 50, tức 2 triệu quỹ đạo; PPO nhỏ: 400 update × 50 × 4 epoch = 80 nghìn quỹ đạo hiệu dụng; PPO lớn: khoảng 2 triệu | Chạy cả hai chế độ nhỏ và lớn; báo theo quỹ đạo hiệu dụng và theo giờ GPU |
| Chọn mô hình | Tốt nhất trên tập validation | Giống |
| Giải mã | Greedy; sampling 100 quỹ đạo lấy lời giải tốt nhất | Giống, kèm đo thời gian |
| Seed | 4 lần chạy (Bảng 6) | Ít nhất 4 seed; báo trung bình và độ lệch |
| OR-Tools | 1800 s mỗi instance cho tập tổng hợp (chú thích Bảng 1); 3600 s cho benchmark (phần văn bản) | Ghi rõ giới hạn dùng; ở cỡ lớn dùng cùng giới hạn và ghi tỷ lệ instance giải tối ưu |
| Phần cứng | Một GPU NVIDIA RTX A40 | Ghi rõ GPU của bạn; chạy lại ReSched trên cùng máy, không so thời gian chéo máy |
| Mô hình | 2 lớp Transformer, 8 head, chiều ẩn 128, Adam, tốc độ học 5e-5 | Báo số tham số PF-UNet và số tham số ReSched |

**Tiêu chí so sánh**

- Gap tính so với OR-Tools ở tập tổng hợp và so với upper bound tốt nhất đã biết ở benchmark. Ở cỡ lớn dùng lời giải tốt nhất trong mọi phương pháp (kể cả OR-Tools) và ghi rõ nguồn.
- Chạy lại ReSched và DANIEL bằng mã gốc, cùng instance và cùng seed sinh dữ liệu, trên cùng máy. Không lấy số từ bài báo cho các cỡ mà bài chưa báo cáo.
- Tập test ở cỡ mới: đề xuất ít nhất 20 instance mỗi cỡ (ReSched dùng 100 ở các cỡ cũ), tăng nếu ngân sách cho phép.
- Dao động giữa các seed của ReSched khoảng 1 điểm phần trăm (Bảng 6), nên chênh lệch dưới ngưỡng đó chưa kết luận được; dùng kiểm định theo cặp trên instance.
- Trường hợp không chạy nổi (hết bộ nhớ hoặc quá thời gian) được ghi lại như một kết quả.

Mình chưa mở repo ReSched, nên tên script, cấu hình và cách sinh dữ liệu cần kiểm tra khi clone.

## 8. Rủi ro

1. Rasterize làm mất chi tiết khi nhiều ứng viên gần nhau; giảm bằng đặc trưng thô trong đầu quyết định và tăng độ phân giải.
2. Tóm tắt phần đuôi mất thông tin máy của operation tương lai (ReSched có chứng minh trạng thái đủ, PF-UNet không); giảm bằng kênh tải tương lai, kiểm chứng bằng ablation.
3. Chi phí U-Net tăng theo số máy M; giảm bằng độ phân giải thấp hoặc chỉ cập nhật tăng dần.
4. Lập luận mật độ hội tụ chỉ đúng cho job độc lập cùng phân phối; có thể hạ xuống thành quan sát thực nghiệm.
5. Novelty phụ thuộc kết quả kiểm tra tiền lệ (SchedDiff, GOAL, CNN-state cho JSSP).

**Rủi ro bổ sung từ Scale-Net:** (a) lợi ích trong phân phối của U-Net có thể rất nhỏ (Scale-Net chỉ cải thiện khoảng 0.05 đến 0.25 điểm phần trăm ở n=50 và n=100), nên đóng góp chính phải đến từ quy mô lớn; (b) chuyển quy mô zero-shot không tự động, cần chứng minh bằng đường cong gap(n); (c) fine-tune tuần tự có thể gây quên, cần báo cáo; (d) phương án dự phòng Graph U-Net gần Scale-Net nên novelty thấp, chỉ dùng làm phương án cuối.

## 9. Mã nguồn và repo liên quan

| Tài nguyên | Địa chỉ | Dùng để | Mức xác minh |
| --- | --- | --- | --- |
| ReSched | github.com/XiangjieXiao/ReSched | Môi trường, reward, đánh giá, baseline | URL theo bài báo, chưa mở |
| DANIEL | github.com/wrqccc/FJSP-DRL | Baseline, môi trường dự phòng, OR-Tools solver | Đã xác minh qua tra cứu |
| HGNN | github.com/songwenas12/fjsp-drl | Baseline | Suy từ repo fork, cần kiểm tra |
| Multi-action DRL | github.com/Lei-Kun/End-to-end-DRL-for-FJSP | Tham khảo | Suy từ repo fork, cần kiểm tra |
| Job Shop Scheduling Benchmark | github.com/ai-for-decision-making-tue/Job\_Shop\_Scheduling\_Benchmark\_Environments\_and\_Instances | Môi trường, luật điều phối, GA, CP-SAT, DANIEL, L2D, instance, FJSP online | Đã xác minh qua tra cứu |
| NCO cho Stochastic FJSP | github.com/ai-for-decision-making-tue/NCO-for-Stochastic-FJSP | Mở rộng sang bất định (SD3) | URL theo bài báo |
| OR-Tools | github.com/google/or-tools | CP-SAT làm mốc | Phổ biến, cần kiểm tra phiên bản |
| Graph U-Nets | github.com/HongyangGao/Graph-U-Nets | Phương án dự phòng | Từ trí nhớ, cần kiểm tra |
| PyTorch Geometric | pytorch-geometric.readthedocs.io | Phương án dự phòng (GNN) | Phổ biến |
| Scale-Net | github.com/lsyysl9711/Scale-Net | Tham khảo U-Net top-k pooling làm baseline E6b; dựa trên POMO-MTL | URL theo bài báo, chưa mở |

Các thành phần PyTorch có sẵn: `torch.nn.functional.grid_sample` (đọc ngược song tuyến tính), `Tensor.scatter_add_` (splat), `torch.nn.GroupNorm`. Chưa tìm thấy repo công khai cho PLAN, CoRS, ReLA trong lượt tra cứu.

## 10. Tài liệu tham khảo (chọn lọc)

- Xiao et al. ReSched, ICLR 2026, arXiv:2603.07020.
- Wang et al. DANIEL, IEEE TNNLS 2024, arXiv:2305.05119.
- Song et al. HGNN, IEEE TII 2023.
- Kwan et al. ReLA, arXiv:2601.03646.
- PLAN, arXiv:2608.03041; CoRS, arXiv:2610.26239 (theo URL tìm được).
- SchedDiff, NeurIPS 2026; GOAL, arXiv:2605.19119.
- Smit et al. NCO cho Stochastic FJSP, AAAI 2025, arXiv:2412.14052.
- Wang et al. MIStar (memory-enhanced improvement heuristics), arXiv:2603.02846.
- Gao & Ji. Graph U-Nets, ICML 2019, arXiv:1905.05178.
- Ronneberger et al. U-Net, MICCAI 2015.
- Lang et al. PointPillars, CVPR 2019; He et al. Mask R-CNN (RoIAlign), ICCV 2017.
- Mao et al. Deep RM (Resource Management with Deep RL), HotNets 2016.
- Reijnen et al. Job Shop Scheduling Benchmark, arXiv:2308.12794.

* Liu, S.; Cao, Z.; Yin, N.; Ong, Y.-S. Scale-Net: A Hierarchical U-Net Framework for Cross-Scale Generalization in Multi-Task Vehicle Routing, AAAI 2026.

*Các tham chiếu không có mã arXiv hoặc năm ở trên được ghi từ trí nhớ hoặc kết quả tra cứu, cần xác minh khi trích dẫn chính thức.*
