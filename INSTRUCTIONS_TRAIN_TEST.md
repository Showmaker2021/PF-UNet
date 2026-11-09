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

Chạy trực tiếp file `train.py` tại thư mục gốc của dự án (`ReSched/`).

### A. Huấn luyện với REINFORCE (Khuyến nghị chạy trước)
* **Thời gian trên M4 Pro:** ~1.5 đến 2 tiếng (2,000 epochs).
* **VRAM / RAM:** ~2.5 GB (rất nhẹ và ổn định).

```bash
# 1. Chạy mặc định trên kích thước 10x5 (Cấu hình chuẩn của bài báo)
python train.py --algo reinforce --model pfunet --size 10x5 --epochs 2000 --batch-size 50

# 2. Hoặc train trên kích thước lớn hơn: 20x10 hoặc 15x10
python train.py --algo reinforce --model pfunet --size 20x10 --epochs 2000 --batch-size 50
python train.py --algo reinforce --model pfunet --size 15x10 --epochs 2000 --batch-size 50

# Hoặc chạy thử nghiệm nhanh (Small budget: 80 epochs, mất ~5-8 phút)
python train.py --algo reinforce --model pfunet --small-budget
```

### B. Huấn luyện với PPO
* **Thời gian trên M4 Pro:** ~7 đến 8 tiếng (2,000 epochs, $K=4$ updates - thích hợp cắm qua đêm).
* **VRAM / RAM:** ~4.5 GB.

```bash
# 1. Chạy mặc định trên kích thước 10x5
python train.py --algo ppo --model pfunet --size 10x5 --epochs 2000 --batch-size 50

# 2. Hoặc train trên kích thước lớn hơn: 20x10 hoặc 15x10
python train.py --algo ppo --model pfunet --size 20x10 --epochs 2000 --batch-size 50

# Hoặc chạy thử nghiệm nhanh (Small budget: 400 epochs, mất ~1.5 tiếng)
python train.py --algo ppo --model pfunet --small-budget
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
- **SD1 10x5** (In-distribution Songwen)
- **SD2 10x5+mix** (Out-of-distribution TNNLS)

### Lệnh chạy đánh giá:
```bash
# Đánh giá checkpoint PF-UNet (REINFORCE)
python test.py --algo reinforce --model pfunet --model-path result/FJSP/<Tên_thư_mục_run>/best_model.pth

# Đánh giá checkpoint PF-UNet (PPO)
python test.py --algo ppo --model pfunet --model-path result/FJSP/<Tên_thư_mục_run>/best_model.pth

# Đánh giá với chế độ Sampling (100 lần lấy nghiệm tốt nhất)
python test.py --algo reinforce --model pfunet --model-path <path_to_model.pth> --inference-type sampling --sample-times 100
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
