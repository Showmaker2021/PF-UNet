# Hướng dẫn Huấn luyện & Đánh giá PF-UNet cho Lập lịch FJSP

Tài liệu này hướng dẫn chi tiết cách chạy **Huấn luyện (Training)** và **Đánh giá (Evaluation/Testing)** mô hình **PF-UNet** (Priority-Field U-Net) hoặc đối chuẩn với **ReSched** (ICLR 2026) trên các bộ benchmark tiêu chuẩn quốc tế.

Hệ thống hỗ trợ tự động:
- **Apple Silicon (M1 / M2 / M3 / M4 Pro / Max)** qua backend Metal GPU (`mps`).
- **NVIDIA GPU** qua CUDA (`cuda`).
- **CPU** fallback tự động nếu không có GPU.

---

## 1. Cài đặt môi trường (Environment Setup)

### Trên máy Mac (Apple Silicon M4 Pro):
Mở Terminal và cài đặt các thư viện phụ thuộc:
```bash
pip install torch torchvision numpy scipy pandas
```
*(Khuyến nghị PyTorch $\ge 2.2$ để tối ưu tăng tốc Metal Performance Shaders - MPS).*

### Trên máy Windows / Linux (NVIDIA CUDA):
```bash
pip install torch torchvision numpy scipy pandas
```

---

## 2. Huấn luyện (Training)

Chạy file `train.py` hoặc `run_all.py` tại thư mục gốc của dự án (`ReSched/`).

### Cách 1: Huấn luyện từng mô hình theo kích thước
Hệ thống tự động đồng bộ siêu tham số theo bài báo gốc (kích thước nhỏ $10\times 5, 20\times 5$ dùng `batch_size=50, episode=1000`; kích thước lớn $15\times 10, 20\times 10$ tự động giảm `batch_size=24, episode=500` theo đúng phụ lục B.2 của ReSched):

```bash
# --- Huấn luyện hệ dữ liệu SD1 (Thời gian [1, 20], ops [0.8m, 1.2m]) ---
python train.py --algo reinforce --model pfunet --dataset SD1 --size 10x5
python train.py --algo reinforce --model pfunet --dataset SD1 --size 20x5
python train.py --algo reinforce --model pfunet --dataset SD1 --size 15x10
python train.py --algo reinforce --model pfunet --dataset SD1 --size 20x10

# --- Huấn luyện hệ dữ liệu SD2 (Thời gian [1, 99], ops = m, mixed machines) ---
python train.py --algo reinforce --model pfunet --dataset SD2 --size 10x5
python train.py --algo reinforce --model pfunet --dataset SD2 --size 20x5
python train.py --algo reinforce --model pfunet --dataset SD2 --size 15x10
python train.py --algo reinforce --model pfunet --dataset SD2 --size 20x10
```

### Cách 2: Huấn luyện tự động toàn bộ (Chạy batch tuần tự qua đêm)
File `run_all.py` sẽ tự động train nối tiếp từng mô hình và lưu riêng biệt:

```bash
# 1. Tự động train tuần tự cả 4 size của SD1 (~7-8 tiếng bằng REINFORCE trên M4 Pro)
python run_all.py --algo reinforce --model pfunet --dataset SD1

# 2. Tự động train tuần tự cả 4 size của SD2
python run_all.py --algo reinforce --model pfunet --dataset SD2

# 3. CHẠY TRỌN VẸN TOÀN BỘ (8 mô hình cho cả SD1 và SD2 để điền đủ Bảng 1)
python run_all.py --algo reinforce --model pfunet --dataset all

# 4. Kiểm thử nhanh toàn bộ luồng pipeline (80 epochs mỗi size)
python run_all.py --algo reinforce --model pfunet --dataset SD1 --small-budget
```

### Bảng tham số của `train.py`:
| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--dataset` | `SD1` | Hệ dữ liệu huấn luyện: `SD1` hoặc `SD2` |
| `--size` | `10x5` | Kích thước bài toán: `10x5`, `20x5`, `15x10`, `20x10` |
| `--algo` | `reinforce` | Thuật toán RL: `reinforce` (chính trong Table 1) hoặc `ppo` |
| `--model` | `pfunet` | Kiến trúc: `pfunet` (mô hình của ta) hoặc `resched` (baseline) |
| `--epochs` | `2000` | Số epoch huấn luyện (mặc định 2000) |
| `--device` | `auto` | Thiết bị tính toán: `auto` (ưu tiên `mps` > `cuda` > `cpu`) |
| `--small-budget` | `False` | Cờ chạy test nhanh (80 epochs cho REINFORCE) |
| `--seed` | `42` | Random seed tái lập kết quả |

---

## 3. Đánh giá Benchmark (Evaluation / Testing)

Sau khi huấn luyện xong, trọng số tốt nhất được lưu tại:
`result/FJSP/<Timestamp_RunName>/best_model.pth`

Bảng 1 trong bài báo ICLR 2026 chia kết quả đánh giá thành **3 phân vùng rõ rệt**:
1. **In-distribution (Cùng phân phối)**: Đánh giá trên tập test cùng kích thước bài toán với mô hình đã học.
2. **Out-of-distribution / OOD (Ngoại suy kích thước lớn)**: Kiểm thử khả năng tổng quát hóa lên các bài toán lớn hơn gấp nhiều lần ($30\times 10$ và $40\times 10$).
3. **Standard Benchmarks (Bộ chuẩn kinh điển)**: Brandimarte và 3 bộ Hurink (rdata, edata, vdata). **Lưu ý: Tác giả chỉ dùng checkpoint huấn luyện trên SD1 để test benchmark chuẩn (không dùng SD2)**.

Đồng thời, mỗi bài toán được báo cáo dưới **2 chế độ giải mã (Decoding Strategy)**:
- **Greedy Decoding** (`--inference-type greedy`): Chọn hành động có xác suất cao nhất tại mỗi bước, tốc độ cực nhanh (vài giây đến 1 phút).
- **Sampling Decoding** (`--inference-type sampling --sample-times 100`): Lấy mẫu ngẫu nhiên 100 quỹ đạo nghiệm rồi lấy nghiệm Makespan tốt nhất, cho chất lượng nghiệm cao nhất.

---

### Bảng tổng hợp ánh xạ Checkpoint vs Mục tiêu đánh giá

| Mục tiêu đánh giá trong Bảng 1 | Dataset Test | Kích thước | Checkpoint cần nạp |
| :--- | :--- | :--- | :--- |
| **In-distribution (SD1)** | SD1 | `10x5`, `20x5`, `15x10`, `20x10` | Checkpoint tương ứng từng bài (`SD1-10x5`, `SD1-20x5`,...) |
| **In-distribution (SD2)** | SD2 | `10x5`, `20x5`, `15x10`, `20x10` | Checkpoint tương ứng từng bài (`SD2-10x5`, `SD2-20x5`,...) |
| **OOD SD1 ($30\times 10, 40\times 10$)** | SD1 | `30x10`, `40x10` | Checkpoint `SD1-10x5` và `SD1-20x10` |
| **OOD SD2 ($30\times 10, 40\times 10$)** | SD2 | `30x10+mix`, `40x10+mix` | Checkpoint `SD2-10x5` và `SD2-20x10` |
| **Benchmark (Brandimarte + Hurink)** | BenchData | Mk01-10, Hurink (r, e, v) | **Duy nhất Checkpoint `SD1-10x5` và `SD1-15x10`** |

---

### Hướng dẫn chi tiết từng lệnh chạy thực tế:

#### Nhóm 1: Đánh giá In-distribution (Chính bài toán vừa train xong)
*Dùng chính checkpoint của bài đó để test:*

```bash
# --- Ví dụ: Đánh giá checkpoint SD1 10x5 ---
# Chế độ Greedy:
python test.py --algo reinforce --model pfunet --model-path result/FJSP/<Folder_SD1_10x5>/best_model.pth --suite sd1 --size 10x5 --inference-type greedy

# Chế độ Sampling (100 lần lấy nghiệm tốt nhất):
python test.py --algo reinforce --model pfunet --model-path result/FJSP/<Folder_SD1_10x5>/best_model.pth --suite sd1 --size 10x5 --inference-type sampling --sample-times 100

# --- Tương tự cho các kích thước khác của SD1 (20x5, 15x10, 20x10) ---
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_20x5.pth> --suite sd1 --size 20x5
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_15x10.pth> --suite sd1 --size 15x10
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_20x10.pth> --suite sd1 --size 20x10

# --- Đánh giá in-distribution cho SD2 ---
python test.py --algo reinforce --model pfunet --model-path result/FJSP/<Folder_SD2_10x5>/best_model.pth --suite sd2 --size 10x5 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path result/FJSP/<Folder_SD2_10x5>/best_model.pth --suite sd2 --size 10x5 --inference-type sampling --sample-times 100
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_20x5.pth> --suite sd2 --size 20x5
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_15x10.pth> --suite sd2 --size 15x10
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_20x10.pth> --suite sd2 --size 20x10
```

---

#### Nhóm 2: Đánh giá Ngoại suy kích thước lớn (Out-of-Distribution - OOD)
*Trong Bảng 1, tác giả dùng checkpoint `10x5` và `20x10` để đánh giá trên hai kích thước lớn là $30\times 10$ và $40\times 10$:*

```bash
# --- 1. Dùng checkpoint SD1 10x5 test OOD SD1 (30x10 và 40x10) ---
# Greedy:
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_10x5.pth> --suite sd1 --size 30x10 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_10x5.pth> --suite sd1 --size 40x10 --inference-type greedy
# Sampling:
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_10x5.pth> --suite sd1 --size 30x10 --inference-type sampling --sample-times 100
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_10x5.pth> --suite sd1 --size 40x10 --inference-type sampling --sample-times 100

# --- 2. Dùng checkpoint SD1 20x10 test OOD SD1 (30x10 và 40x10) ---
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_20x10.pth> --suite sd1 --size 30x10 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_20x10.pth> --suite sd1 --size 40x10 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_20x10.pth> --suite sd1 --size 30x10 --inference-type sampling --sample-times 100
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_20x10.pth> --suite sd1 --size 40x10 --inference-type sampling --sample-times 100

# --- 3. Dùng checkpoint SD2 10x5 và 20x10 test OOD SD2 (30x10 và 40x10) ---
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_10x5.pth> --suite sd2 --size 30x10 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_10x5.pth> --suite sd2 --size 40x10 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_20x10.pth> --suite sd2 --size 30x10 --inference-type greedy
python test.py --algo reinforce --model pfunet --model-path <path_to_SD2_20x10.pth> --suite sd2 --size 40x10 --inference-type greedy
```

---

#### Nhóm 3: Đánh giá trên Benchmark kinh điển (Brandimarte + Hurink)
*Trong Bảng 1, tác giả nạp checkpoint huấn luyện từ **SD1-10x5** và **SD1-15x10** để chạy trên 4 tập dữ liệu chuẩn quốc tế:*
- `Brandimarte` (10 instances Mk01-Mk10)
- `Hurink_rdata` (66 instances)
- `Hurink_edata` (66 instances)
- `Hurink_vdata` (66 instances)

```bash
# --- 1. Đánh giá bằng Checkpoint SD1 10x5 ---
# Chế độ Greedy (chạy 4 benchmark mất khoảng ~1-2 phút):
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_10x5.pth> --suite benchmark --inference-type greedy

# Chế độ Sampling 100 mẫu (kết quả Makespan và Gap đẹp nhất):
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_10x5.pth> --suite benchmark --inference-type sampling --sample-times 100

# --- 2. Đánh giá bằng Checkpoint SD1 15x10 ---
# Chế độ Greedy:
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_15x10.pth> --suite benchmark --inference-type greedy

# Chế độ Sampling 100 mẫu:
python test.py --algo reinforce --model pfunet --model-path <path_to_SD1_15x10.pth> --suite benchmark --inference-type sampling --sample-times 100
```

---

#### Nhóm 4: Lệnh quét nhanh toàn diện (Batch Scan)
Nếu bạn muốn 1 lệnh kiểm thử quét qua hàng loạt datasets:

```bash
# Quét toàn bộ 4 Benchmark:
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite benchmark

# Quét tất cả 6 kích thước của SD1 (10x5, 15x10, 20x5, 20x10, 30x10, 40x10):
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite sd1

# Quét tất cả 6 kích thước của SD2:
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite sd2

# Quét FULL toàn bộ 16 datasets có trong Bảng 1:
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite all
```

Kết quả Makespan và Optimality Gap (%) so sánh với cận trên tối ưu (Upper Bound - UB / OR-Tools) sẽ được in trực tiếp lên màn hình và tự động lưu vào file nhật ký `log.txt`.

---

## 4. Cấu trúc thư mục đầu ra (Output Structure)

Khi huấn luyện, kết quả sẽ tự động lưu trong thư mục `result/`:
```text
result/
└── FJSP/
    └── 20261109_xxxxxx_SD1-10x05-PFUNET-REINFORCE/
        ├── log.txt                 # Nhật ký điểm Makespan & Loss từng epoch
        ├── checkpoint.pth          # Checkpoint đầy đủ (để resume nếu bị ngắt quãng)
        ├── best_model.pth          # Trọng số tối ưu nhất (dùng nạp vào test.py)
        └── src/                    # Bản sao lưu snapshot mã nguồn tại thời điểm chạy
```

---

## 5. Xuất hình minh hoạ Concept cho Paper (Figures for Paper)

Để xuất lại hình minh họa kiến trúc 2 nhánh Priority-Field U-Net đạt tiêu chuẩn xuất bản quốc tế (vector PDF và 300 DPI PNG):
```bash
python PPO/pfunet/figures/plot_pfunet_concept.py
```
File được xuất ra tại:
- Vector PDF: `PPO/pfunet/figures/figures/pfunet_overall_concept.pdf`
- Hình PNG 300 DPI: `PPO/pfunet/figures/figures/pfunet_overall_concept.png`
