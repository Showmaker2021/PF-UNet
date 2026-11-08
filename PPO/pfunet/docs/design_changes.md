# Nhật ký thay đổi thiết kế (design_changes.md)

Tài liệu này ghi lại mọi thay đổi thiết kế giữa PF-UNet và đề cương gốc hoặc ReSched.

## Thay đổi ban đầu (từ Guide v3)
- **D1. Thêm, không thay:** ReSched giữ nguyên làm baseline trong `PPO/`. PF-UNet là package con `PPO/pfunet/`. Lớp `SchedulingModel` của PF-UNet giữ nguyên chữ ký giao diện.
- **D2. Suy nhóm job từ `dependency_o`:** Giữ nguyên tuple 8 phần tử, không đổi interface đầu vào của môi trường. Cần kiểm chứng chiều ma trận bằng `test_dependency_direction`.
- **D3. Tiến độ công việc:** Thay $n_j / N_{j0}$ bằng $n_j / \max_j n_j$ (kênh 5 và đặc trưng thô thứ 7) do mô hình không nhận $N_{j0}$ ban đầu.
- **D4. Slot job = Head:** Số slot trong batch là $J = \max_b J_a$. Không cần padding máy vì số máy $M$ cố định trong batch.
- **D5. Khôi phục không gian hành động bằng `scatter_add_`:** Chuyển logits từ `[B, M, J]` thành xác suất `prob [B, N_rem, M]` trong không gian môi trường bằng `scatter_add_`, tránh dùng `scatter_` để không bị slot đệm ghi đè 0. Sao chép nguyên văn khối chọn hành động của ReSched.
- **D6. Critic:** Dự đoán giá trị $V$ qua MLP từ trung bình theo máy của vector bottleneck $g$ sau attention liên máy.


## Kết quả kiểm thử thực nghiệm (Phases P0 - P5)

### P1: Đánh giá Baseline ReSched
- Tải checkpoint gốc `ckpt/PPO/SD1-10x05.pth`, chạy greedy evaluation trên toàn bộ benchmark:
  - Brandimarte: 185.6000 (Gap 10.34%) - Trùng khớp 100% Bảng 10 ReSched paper
  - Hurink_rdata: 1028.7250 (Gap 10.60%) - Trùng khớp 100% Bảng 10 ReSched paper
  - Hurink_edata: 1197.8500 (Gap 16.25%) - Trùng khớp 100% Bảng 10 ReSched paper
  - Hurink_vdata: 975.1000 (Gap 6.42%) - Trùng khớp 100% Bảng 10 ReSched paper
  - SD1 10x5: 106.2600 (Gap 11.48%) - Trùng khớp 100% Bảng 10 ReSched paper
  - SD2 10x5+mix: 372.8000 (Gap 14.24%) - Trùng khớp 100% Bảng 10 ReSched paper
- Đã ghi lại makespan của 20 instance cố định đầu tiên của SD1 10x5 vào `baseline_20_makespans.npy`.

### P2: Reference Simulator
- Triển khai `PPO/pfunet/ref_sim.py` (`RefSim`, `to_ref`, `ref_obs`).
- Kiểm thử `tests/test_ref_parity.py`: Đạt **1000/1000** sequence ngẫu nhiên trùng khớp chính xác 100% cả makespan và vector available time của máy (`m_AT` vs `sim.A`).

### P3: Observation Builder
- Kiểm thử `tests/test_dependency_direction.py`: Kiểm chứng giả thuyết D2 trên **19,888** batch states reachable từ rollout ngẫu nhiên trên SD1 và FJSP chuẩn. 100% khớp `dependency_o[head] == same_job[head]`.
- Triển khai `build_pf_obs` vector hóa hoàn toàn trong `PPO/pfunet/modules.py`.
- Kiểm thử `tests/test_obs_builder.py`: 200 trạng thái ngẫu nhiên đạt parity tuyệt đối (`atol=1e-5`) so với `ref_obs`, toạ độ luôn nằm trong `[0, 1]`, số head bằng số job còn sống.

### P4: U-Net, Rasterizer, Readout & Policy Interface
- Triển khai `rasterize`, `rasterize_obs`, `readout`, `ConvBlock`, `UNet2D`, `MachineAttention`, `MLP` trong `PPO/pfunet/modules.py`.
- Triển khai `SchedulingModel` trong `PPO/pfunet/model.py`.
- Kiểm thử `tests/test_rasterize.py`: Đạt bảo toàn khối lượng (mass conservation), one-hot tại tâm pixel, chia đều 0.25 tại trung điểm, readout bản đồ hằng số trả về hằng số.
- Kiểm thử `tests/test_invariance.py`: Đạt tương đương hoán vị job, hoán vị máy, và bất biến tỷ lệ thời gian.
- Kiểm thử `tests/test_policy_interface.py`: Hỗ trợ greedy, sampling, teacher_forcing; tổng xác suất bằng 1; backward pass thông suốt đến tất cả 104 tensor tham số; VRAM đỉnh ~921 MB trên RTX 3050 Laptop GPU.

### P5: Hook & Regression & Overfitting
- Hook sạch sẽ tại `PPO/configs/fjsp.py` và `PPO/SchedulingRunner.py`. Mặc định `model_type='resched'`. Khi chọn `model_type='pfunet'`, khởi tạo `pfunet.model.SchedulingModel`.
- Kiểm thử `tests/test_baseline_unchanged.py`: Xác nhận chạy greedy mô hình gốc ReSched trên 20 instance cố định cho makespan giống từng bit với `baseline_20_makespans.npy`.
- Kiểm thử `tests/test_overfit_one.py`: Mô hình học tốt trên 1 instance SD1 10x5 (instance 0, OR-Tools optimal = 109.0):
  - Makespan greedy ban đầu: 235.0.
  - Ngay trong những epoch đầu, makespan giảm xuống 122.0 (+11.9% gap).
  - Sample average loss và makespan giảm đều và ổn định.
  - VRAM ổn định ở mức ~3.8 GB, thời gian mỗi epoch ~10.7s trên GPU.


### Tích hợp REINFORCE (Dual Algorithm Support)
- **Hỗ trợ đa thuật toán trong `SchedulingModel`:**
  - Thêm cờ `rl_algo`: hỗ trợ `'ppo'` (mặc định) và `'reinforce'`.
  - Trong chế độ `'reinforce'`: Mạng Critic không được khởi tạo (`self.critic = None`), giúp tiết kiệm tham số và VRAM.
  - Chữ ký hàm `forward`:
    - Khi `rl_algo == 'ppo'`: Trả về 4-tuple `([seq_id, machine_id], prob_selected, value, prob_distribution)`.
    - Khi `rl_algo == 'reinforce'`: Trả về đúng 2-tuple `([seq_id, machine_id], prob_selected)` khớp hoàn hảo với `REINFORCETrainer.py` và `SchedulingEvaluator.py` của nhánh REINFORCE.
- **Hook tích hợp sạch vào `REINFORCE/`:**
  - `REINFORCE/configs/fjsp.py`: Khai báo khối `PFUNET_PARAMS` với `'rl_algo': 'reinforce'`. Mặc định `model_params` vẫn là `'resched'`.
  - `REINFORCE/SchedulingRunner.py`: Nhúng hook chọn `ModelCls = PFUNetModel if model_type == 'pfunet' else Model` và cấu hình `torch.set_default_device(self.device)`.
- **Kiểm thử nghiệm thu (`tests/test_reinforce.py`):**
  - Khởi tạo mô hình ở chế độ REINFORCE không có Critic.
  - Forward pass trả về đúng 2-tuple `(action, prob)`.
  - Đồ thị tính toán lan truyền ngược sạch sẽ qua toàn bộ tham số U-Net.
  - Chạy 5 epoch huấn luyện GPU thực tế với `REINFORCETrainer`: Hoạt động trơn tru 100%, loss giảm ổn định và không gặp bất kỳ lỗi xung đột device hay gradient nào.
