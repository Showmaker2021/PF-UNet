# Hướng dẫn Huấn luyện & Đánh giá PF-UNet cho Lập lịch FJSP

Tài liệu này hướng dẫn chi tiết cách chạy **Huấn luyện (Training)** và **Đánh giá (Evaluation/Testing)** mô hình **PF-UNet** (Priority-Field U-Net) hoặc so sánh với **ReSched** trên các bộ benchmark tiêu chuẩn.

Hệ thống hỗ trợ tự động:
- **Apple Silicon (M1 / M2 / M3 / M4 Pro / Max)** qua backend Metal GPU (`mps`).
- **NVIDIA GPU** qua CUDA (`cuda`).
- **CPU** fallback nếu không có GPU.

---

## 1. Cài đặt môi trường (Environment Setup)

### Trên máy Mac (Apple Silicon M4 Pro):
Mở terminal và cài đặt các thư viện cần thiết:
```bash
pip install torch torchvision numpy scipy pandas
```
*(Khuyến nghị PyTorch $\ge 2.2$ để tối ưu Metal Performance Shaders).*

### Trên máy Windows / Linux (NVIDIA CUDA):
```bash
pip install torch torchvision numpy scipy pandas
```

---

## 2. Huấn luyện (Training)

Chạy trực tiếp file `train.py` hoặc `run_all.py` tại thư mục gốc của dự án (`ReSched/`).

### Cách 1: Huấn luyện từng kích thước bài toán
Hệ thống tự động khớp cấu hình theo bài báo (size nhỏ $10\times 5, 20\times 5$ dùng `batch_size=50`, size lớn $15\times 10, 20\times 10$ tự động giảm `batch_size=24` như trong phụ lục B.2 của ReSched):

```bash
# Huấn luyện mô hình chuẩn 10x5 (Khuyến nghị chạy trước, ~1.5 đến 2 tiếng trên M4 Pro)
python train.py --algo reinforce --model pfunet --dataset SD1 --size 10x5

# Huấn luyện các kích thước khác của SD1
python train.py --algo reinforce --model pfunet --dataset SD1 --size 20x5
python train.py --algo reinforce --model pfunet --dataset SD1 --size 15x10
python train.py --algo reinforce --model pfunet --dataset SD1 --size 20x10

# Huấn luyện trên hệ dữ liệu SD2
python train.py --algo reinforce --model pfunet --dataset SD2 --size 10x5
```

### Cách 2: Huấn luyện tự động toàn bộ kích thước (Cắm qua đêm)
File `run_all.py` cho phép tự động train tuần tự từng mô hình một:

```bash
# 1. Tự động train tuần tự cả 4 size của SD1 (~7-8 tiếng bằng REINFORCE)
python run_all.py --algo reinforce --model pfunet --dataset SD1

# 2. Tự động train tuần tự cả 4 size của SD2
python run_all.py --algo reinforce --model pfunet --dataset SD2

# 3. CẮM FULL CẢ SD1 VÀ SD2 (Toàn bộ 8 models cho trọn vẹn Bảng 1)
python run_all.py --algo reinforce --model pfunet --dataset all

# Hoặc test nhanh toàn bộ luồng trước khi chạy thật (80 epochs mỗi size)
python run_all.py --algo reinforce --model pfunet --dataset SD1 --small-budget
```

### Các tùy chọn dòng lệnh của `train.py`:
| Cờ (Flag) | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--size` | `10x5` | Kích thước bài toán huấn luyện: `10x5`, `15x10`, `20x5`, `20x10` |
| `--algo` | `reinforce` | Thuật toán học tăng cường: `reinforce` hoặc `ppo` |
| `--model` | `pfunet` | Kiến trúc mô hình: `pfunet` (mô hình của ta) hoặc `resched` (baseline) |
| `--epochs` | `2000` | Số epoch huấn luyện |
| `--batch-size` | `50` | Kích thước batch (khớp 100% ReSched gốc) |
| `--mini-batch-size` | `50` | Kích thước mini-batch cho PPO update |
| `--k-epochs` | `4` | Số lượt update lặp lại của PPO |
| `--device` | `auto` | Thiết bị tính toán: `auto` (tự chọn `mps` > `cuda` > `cpu`), hoặc ép `cuda`, `mps`, `cpu` |
| `--small-budget` | `False` | Cờ chạy nhanh (80 epochs cho REINFORCE, 400 epochs cho PPO) |
| `--seed` | `42` | Random seed tái lập kết quả |
| `--run-name` | Tự sinh | Tên folder lưu log và checkpoint trong `result/` |

---

## 3. Đánh giá Benchmark (Evaluation / Testing)

Sau khi huấn luyện xong, mô hình tốt nhất sẽ được lưu tại:
`result/FJSP/<Timestamp_RunName>/best_model.pth`

Dùng file `test.py` để đánh giá trên toàn bộ 6 bộ benchmark tiêu chuẩn:
- **Brandimarte** (10 bài toán kinh điển)
- **Hurink_r, Hurink_e, Hurink_v** (bộ dữ liệu phức tạp)
### Các tập dữ liệu kiểm thử hỗ trợ (`--suite`):
- `--suite benchmark` (Mặc định cho benchmark công khai): Gồm 4 bộ dữ liệu kinh điển:
  - **Brandimarte** (Mk01 - Mk10)
  - **Hurink_rdata** (66 instances)
  - **Hurink_edata** (66 instances)
  - **Hurink_vdata** (66 instances)
- `--suite sd1`: Toàn bộ 6 bài SD1 (10x5, 15x10, 20x5, 20x10, 30x10, 40x10).
- `--suite sd2`: Toàn bộ 6 bài SD2 (10x5+mix, 15x10+mix, 20x5+mix, 20x10+mix, 30x10+mix, 40x10+mix).
- `--suite all`: Kiểm thử toàn diện toàn bộ 16 tập dữ liệu có trong Bảng 1 paper ReSched.
- `--size <10x5|15x10|20x5|20x10|30x10|40x10>`: Lọc chính xác 1 kích thước cần test trong SD1 hoặc SD2.
- `--dataset-path <đường_dẫn>`: Trỏ trực tiếp đến 1 hoặc nhiều thư mục dữ liệu bất kỳ.

### Lệnh chạy đánh giá:
```bash
# 1. Đánh giá trên bộ 4 Benchmark chuẩn (Brandimarte + Hurink r, e, v)
python test.py --algo reinforce --model pfunet --model-path result/FJSP/<Tên_thư_mục_run>/best_model.pth --suite benchmark

# 2. Đánh giá in-distribution trên bài huấn luyện (Ví dụ SD1 10x5)
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite sd1 --size 10x5

# 3. Đánh giá OOD trên SD1 kích thước lớn (30x10, 40x10)
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite sd1

# 4. Đánh giá toàn bộ 16 datasets (Bảng 1 Table 1)
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite all

# 5. Đánh giá với chế độ Sampling (100 lần lấy nghiệm tốt nhất)
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --suite benchmark --inference-type sampling --sample-times 100
```

Kết quả Makespan và Optimality Gap (%) sẽ được in trực tiếp ra màn hình và ghi vào file `log.txt`.

---

## 4. Cấu trúc thư mục đầu ra (Output Structure)

Khi huấn luyện, hệ thống tự động sinh thư mục lưu trữ trong `result/`:
```text
result/
└── FJSP/
    └── 20261109_xxxxxx_SD1-10x05-PFUNET-REINFORCE/
        ├── log.txt                 # Nhật ký Makespan & Loss từng epoch
        ├── checkpoint.pth          # Checkpoint đầy đủ (gồm optimizer & scheduler để resume)
        ├── best_model.pth          # Trọng số tốt nhất (dùng để test và nộp bài báo)
        └── src/                    # Bản sao lưu mã nguồn tại thời điểm chạy
```

---

## 5. Vẽ hình minh hoạ Concept cho Paper (Figures for Paper)

Để xuất lại hình minh hoạ kiến trúc PF-UNet chuẩn xuất bản (Nature / NeurIPS style):
```bash
python PPO/pfunet/figures/plot_pfunet_concept.py
```
Hình ảnh xuất ra gồm file PNG và file PDF chất lượng cao tại:
`PPO/pfunet/figures/figures/pfunet_overall_concept.pdf`
