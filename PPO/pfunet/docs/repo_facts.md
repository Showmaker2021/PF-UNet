# Báo cáo Xác minh Mã nguồn ReSched (repo_facts.md)

Tài liệu này xác minh chi tiết toàn bộ các khía cạnh kỹ thuật cốt lõi trong codebase ReSched dựa trên việc đọc và phân tích mã nguồn thực tế kèm định vị chính xác `tệp:dòng` (file:line).

---

## 1. Hợp đồng Mô hình – Trainer (Model–Trainer Contract)

### 1.1. Đầu vào của mô hình
- **Kiểu dữ liệu:** Mô hình nhận một **tuple gồm 8 tensor rời** (KHÔNG PHẢI dictionary / từ điển).
- **Vị trí định nghĩa:** `PPO/SchedulingModel.py:8-45` (hàm `build_model_input`) và unpack tại `PPO/SchedulingModel.py:70-72`.
- **Cấu trúc tuple truyền vào:**
  ```python
  (min_duration, duration, o_at, m_at, dependency_o, dependency_m, action_mask, rev_pos) = state
  ```
  Khi gọi trong huấn luyện có thể truyền thêm tham số `action=(seq, machine)` để ép bước chọn trong chế độ `teacher_forcing` (`PPO/SchedulingModel.py:70, 107-108`).

### 1.2. Đầu ra của mô hình
- **Trong PPO (`PPO/SchedulingModel.py:112`):** Trả về 4 phần tử:
  ```python
  return [seq_id, machine_id], prob[torch.arange(prob.shape[0]), seq_id, machine_id], value.squeeze(), prob
  ```
  1. `[seq_id, machine_id]`: List gồm 2 tensor 1D kích thước `(B,)`, biểu diễn chỉ số hành động (index của operation trong chuỗi còn lại và index của machine).
  2. `prob_selected`: Tensor 1D kích thước `(B,)`, kiểu `torch.float32`. Đây là **xác suất** (probability đã qua `F.softmax`, KHÔNG PHẢI logit) của hành động được chọn.
  3. `value`: Tensor 1D kích thước `(B,)`, kiểu `torch.float32`. **CÓ giá trị $V(s)$** được dự đoán từ mạng Critic (`PPO/SchedulingModel.py:66, 95`).
  4. `prob`: Tensor 3D kích thước `(B, N_rem, M)`, kiểu `torch.float32`. Phân phối xác suất đầy đủ của toàn bộ không gian hành động sau softmax (các vị trí bị mask có xác suất bằng 0).
- **Trong REINFORCE (`REINFORCE/SchedulingModel.py:100`):** Chỉ trả về 2 phần tử: `[seq_id, machine_id]` và `prob_selected` (không có Critic, không có giá trị V).

### 1.3. Các phương thức của mô hình được gọi
- **Trong `PPO/PPOTrainer.py`:**
  - `self.model.parameters()`: `PPO/PPOTrainer.py:18` (khởi tạo optimizer).
  - `self.model.state_dict()`: `PPO/PPOTrainer.py:32, 36` (sao chép sang `model_old` và lưu checkpoint).
  - `self.model_old.load_state_dict()`: `PPO/PPOTrainer.py:36` (đồng bộ trọng số sang mạng rollout).
  - `self.model_old.eval()`: `PPO/PPOTrainer.py:37` (đặt chế độ rollout sampling).
  - `self.model_old.set_decode_type('sampling')`: `PPO/PPOTrainer.py:38`.
  - `self.model_old(state)` (gọi `__call__` / `forward`): `PPO/PPOTrainer.py:53`.
  - `self.model.train()`: `PPO/PPOTrainer.py:93` (chuyển chế độ train).
  - `self.model.set_decode_type('teacher_forcing')`: `PPO/PPOTrainer.py:94`.
  - `self.model(state, action=action)` (gọi `__call__` / `forward`): `PPO/PPOTrainer.py:117`.
- **Trong `PPO/SchedulingEvaluator.py`:**
  - `model.eval()`: `PPO/SchedulingEvaluator.py:11`.
  - `model.set_decode_type('greedy' if ... else 'sampling')`: `PPO/SchedulingEvaluator.py:12`.
  - `model(state)` (gọi `__call__` / `forward`): `PPO/SchedulingEvaluator.py:23`.
- **Trong `PPO/SchedulingRunner.py`:**
  - `Model(**model_params)` (khởi tạo): `PPO/SchedulingRunner.py:44`.
  - `self.model.load_state_dict()`: `PPO/SchedulingRunner.py:167, 204, 282`.
  - `self.model.state_dict()`: `PPO/SchedulingRunner.py:293, 303`.

---

## 2. Tensor Trạng thái (State Tensors)

Quy ước kích thước:
- $B$: `batch_size`.
- $N_{\text{rem}}$: Số lượng operation còn lại tại bước hiện tại ($N_{\text{rem}} = \text{seq\_len} - \text{decided\_len}$).
- $M$: Số lượng máy (`num_machines`).

| Tên Tensor | Hình dạng (Shape) | Kiểu dữ liệu (dtype) | Tệp và dòng định nghĩa |
| :--- | :--- | :--- | :--- |
| `duration` | `(B, N_rem, M)` | `torch.float32` | `PPO/SchedulingModel.py:28, 38` |
| `min_duration` | `(B, N_rem)` | `torch.float32` | `PPO/SchedulingModel.py:29, 37` |
| `m_AT` | `(B, M)` | `torch.float32` | `PPO/SchedulingModel.py:25, 40` |
| `o_AT` | `(B, N_rem)` | `torch.float32` | `PPO/SchedulingModel.py:26, 39` |
| `position_rev` | `(B, N_rem)` | `torch.int32` (`torch.int`) | `PPO/SchedulingModel.py:9, 44` |
| `action_mask` | `(B, N_rem, M)` | `torch.bool` | `PPO/SchedulingModel.py:34, 43` |
| `o2o_mask` (`dependency_o`) | `(B, N_rem, N_rem)` | `torch.bool` | `PPO/SchedulingModel.py:31, 41` |
| `o2m_mask` (`dependency_m`) | `(B, N_rem, M)` | `torch.bool` | `PPO/SchedulingModel.py:32, 42` |

### Giá trị "không tương thích" của duration:
- **Trong ma trận duration ban đầu:** Giá trị máy không tương thích được đặt là **0** (`PPO/SchedulingEnvironment.py:171, 311` và `PPO/SchedulingGenerator.py:100, 156`).
- **Khi đưa vào mô hình:** Sau khi chia cho `scaler`, giá trị không tương thích vẫn là **0.0** (`PPO/SchedulingModel.py:28`).
- **Trong action mask:** Các vị trí này có `o2m_mask == False`, do đó bị triệt tiêu thành `False` trong `action_mask` (`PPO/SchedulingEnvironment.py:173, 251`).
- **Khi tính Makespan / Greedy cuối:** Môi trường chuyển 0 thành vô cùng bằng `np.where(duration == 0, np.inf, duration)` (`PPO/SchedulingEnvironment.py:100, 183`).

---

## 3. Quản lý Job ID của từng Operation

- **Đưa vào mô hình:** **KHÔNG CÓ tensor job_id nào được truyền vào mô hình** (`PPO/SchedulingModel.py:36-45, 71`). Mô hình hoàn toàn không nhận diện job ID trực tiếp mà học quan hệ qua `o2o_mask` và RoPE `rev_pos`.
- **Lưu trữ phía môi trường:** Môi trường lưu trữ trực tiếp job ID của các operation còn lại qua thuộc tính `self.state.job_idx` (kích thước `(B, N_rem)`, numpy array) (`PPO/SchedulingEnvironment.py:150, 301`).
- **Suy luận từ `o2o_mask`:** Có thể suy ra các operation thuộc cùng một job từ `o2o_mask` (hay `connection`), vì `o2o_mask` kết nối một operation với chính nó và các operation đứng sau của cùng một job (`PPO/SchedulingGenerator.py:180-200`, `PPO/SchedulingEnvironment.py:123-125`).
- **Cập nhật qua `delete_mask`:**
  - Vị trí mã nguồn: `PPO/SchedulingEnvironment.py:294-301`.
  - Cơ chế:
    ```python
    delete_mask = np.ones(self.problem.duration.shape[:2], dtype=bool)
    del_idx = np.array(self.solution.jo_idx).T
    delete_mask[batch_idx[:, None], del_idx] = False

    decided_len = del_idx.shape[-1]
    self.state.job_idx = self.problem.job_idx[delete_mask].reshape(batch_size, seq_len - decided_len)
    ```
    Mỗi khi một operation được chọn, chỉ số của nó bị đánh dấu `False` trong `delete_mask`. Sau đó, `self.problem.job_idx` ban đầu được trích xuất bằng boolean indexing `delete_mask` và reshape về kích thước `(B, N_initial - decided_len)`.

---

## 4. Ý nghĩa của `o_AT` với Operation Không Phải Head

- **Cơ chế cập nhật trong môi trường:**
  - `available_time_o` ban đầu khởi tạo bằng **0** cho tất cả các operation (`PPO/SchedulingEnvironment.py:161, 178`).
  - Khi một operation hoàn thành tại thời điểm $et$, môi trường **chỉ cập nhật** cho duy nhất operation kế tiếp trực tiếp (`first_succ`) của cùng job đó:
    ```python
    self.state.available_time_o[b[succ_exist], first_succ[succ_exist]] = et[succ_exist]
    ```
    (`PPO/SchedulingEnvironment.py:276`).
  - Các operation đứng sau nữa (chưa tới lượt) **vẫn giữ nguyên giá trị mặc định là 0** (`PPO/SchedulingEnvironment.py:307`).
- **Xử lý trong mô hình (`PPO/SchedulingModel.py:74-76`):**
  ```python
  pending_mask = action_mask.sum(dim=-1) > 0
  operation_embedding = self.duration_embedding(min_duration.unsqueeze(-1))
  operation_embedding[pending_mask] += self.oat_embedding(o_at.unsqueeze(-1))[pending_mask]
  ```
  -> **Mô hình chỉ cộng `oat_embedding` vào các operation thỏa mãn `pending_mask` (tức operation head)**. Các operation chưa tới lượt (`~pending_mask`) hoàn toàn không dùng đến `o_at`; embedding của chúng chỉ gồm `duration_embedding(min_duration)`. Nhận định *"chỉ cần o_AT của operation head"* là chính xác 100%.

---

## 5. Xác định Operation Head và Tính Đơn vị theo Job

- **Điều kiện Operation Head:**
  - **Đúng**, operation head chính là các hàng có `action_mask[b, o, :].any() == True`.
  - Căn cứ: `action_mask = o2m_mask * (dependency.sum(1) == 0)[..., None]` (`PPO/SchedulingEnvironment.py:173, 250-251`).
  - Một operation có `dependency.sum(1) == 0` nghĩa là không còn tiền nhiệm nào chưa hoàn thành trong đồ thị còn lại. Vì mỗi operation hợp lệ luôn có ít nhất một máy tương thích, nên hàng này có ít nhất một giá trị `True` trên chiều máy.
- **Tính duy nhất theo Job:**
  - **Đúng, mỗi job có tối đa một hàng như vậy.**
  - Lý do: Trong bài toán Job Shop, các công đoạn trong cùng một job có quan hệ phụ thuộc tuyến tính nghiêm ngặt ($O_1 \to O_2 \to \dots$). Tại mọi thời điểm, chỉ có duy nhất công đoạn đầu tiên chưa hoàn thành của job là có số tiền nhiệm còn lại bằng 0. Khi toàn bộ công đoạn của job hoàn tất, job đó có 0 hàng.

---

## 6. Xử lý Số Operation Khác nhau giữa các Job (SD2) và Đồng bộ Batch

- **Cơ chế của môi trường (Không dùng padding):**
  - Môi trường **không sử dụng padding hay mask độ dài**.
  - Khi đọc dữ liệu kiểm thử SD2 (`PPO/configs/fjsp.py:154-162, 226-255`), các instance được nhóm theo kích thước của ma trận `duration` thông qua `assign_group_ids` và `group_and_batch_by_idx`.
  - Mỗi batch chỉ chứa các instance có cùng kích thước `(N, M)`, tức **cùng tổng số operation**.
  - Khi sinh dữ liệu ngẫu nhiên lúc huấn luyện (`PPO/SchedulingGenerator.py:38-44` hoặc `PPO/SD1FJSPGenerator.py:54-58`), cấu trúc số operation mỗi job được sinh một lần cho toàn batch và lặp lại (`tile`) cho tất cả các instance.
- **Đồng bộ bước kết thúc:**
  - **CÁC INSTANCE TRONG MỘT BATCH LUÔN KẾT THÚC Ở CÙNG MỘT BƯỚC CHÍNH XÁC.** Vì mọi instance trong batch có cùng tổng số operation ban đầu $N$, và tại mỗi bước tuần tự môi trường xử lý đồng thời 1 operation cho mỗi instance ($N \to N-1 \to \dots \to 1$).

---

## 7. Bước Cuối Tự động (Last Step Greedy)

- **Vị trí mã nguồn:** `PPO/SchedulingEnvironment.py:98-105` (trong phương thức `step()`):
  ```python
  if self.state.action_mask.shape[1] == 1:
      duration = self.state.duration.squeeze(axis=1)
      mask_duration = np.where(duration == 0, np.inf, duration)
      finish_time = self.state.m_AT + mask_duration
      machine_idx = np.argmin(finish_time, axis=1)
      action = (np.zeros(self.batch_size, dtype=int), machine_idx)
      _ = self._step(action)
      ...
      done = True
  ```
- **Có tính vào reward không:**
  - **KHÔNG.**
  - `step_reward = self._step(action)` ở dòng 95 tính reward cho bước do mô hình ra quyết định.
  - Ở bước cuối tự động tại dòng 104: `_ = self._step(action)` kết quả trả về bị gán vào biến bỏ qua `_`.
  - Dòng 115 chỉ trả về: `return self.state, step_reward, done` (chính là reward của bước áp chót).

---

## 8. Bộ Sinh Dữ liệu (Generators)

- **Cỡ tùy ý (ví dụ 100×10, 200×5):**
  - **`SD1FJSPGenerator.py` (`PPO/SD1FJSPGenerator.py:17-38`):** Sinh được cỡ tùy ý theo `num_jobs` và `num_mas`. Số lượng operation mỗi job tự động nội suy theo quy chuẩn SD1 trong đoạn $[0.8 \times \text{num\_mas}, 1.2 \times \text{num\_mas}]$.
  - **`SchedulingGenerator.py` (`PPO/SchedulingGenerator.py:126-152`):** Class `FlexibleJobShopProblemGenerator` sinh được cỡ tùy ý cho FJSP tổng quát với các tham số linh hoạt.
  - **Về SD2:** Trong mã nguồn **không có generator riêng mang tên SD2Generator**. Dữ liệu SD2 được lưu sẵn dưới dạng file benchmark trong thư mục `data/FJSP/TNNLS/SD2/`. Tuy nhiên, `FlexibleJobShopProblemGenerator` có thể mô phỏng bài toán có số operation mỗi job biến thiên nếu cấu hình `min_operation` và `max_operation`.
- **Các tham số điều khiển:**
  - Trong `SD1FJSPGenerator.py:17`: `num_jobs`, `num_mas`, `nums_ope`, `path`, `flag_same_opes`, `flag_doc`.
  - Trong `SchedulingGenerator.py:126-139`: `num_jobs`, `num_machines`, `num_operations`, `min_operation`, `max_operation`, `min_processing_time`, `max_processing_time`, `min_eligible_ma_per_op`, `max_eligible_ma_per_op`.

---

## 9. Lệnh Chạy Chuẩn trong README.md

Nguyên văn phần **Quick Start** từ `README.md:20-77`:

```markdown
## Quick Start

### 1. Choose an Algorithm

Run the code inside the corresponding algorithm directory:

```bash
cd REINFORCE
python SchedulingMain.py
```

or:

```bash
cd PPO
python SchedulingMain.py
```

### 2. Choose a Problem Type

Modify the following line in `SchedulingMain.py`:

```python
PROBLEM = 'fjsp'   # 'fjsp', 'jssp', 'ffsp'
```

The program will then automatically load the matching configuration file from `configs/`.

### 3. Test an Existing Model

To evaluate a pretrained model, check the corresponding `configs/*.py` file:

```python
runner_params = {
    'test_only': True,
    'checkpoint': None,
    'model_path': '...'
}
```

Where:

- `test_only=True` means training is skipped and the model is evaluated directly
- `model_path` points to the `.pth` file to load
- `checkpoint` can be used to resume from an existing experiment directory

### 4. Train a New Model

To train a model, update the configuration as follows:

```python
runner_params = {
    'test_only': False,
    'checkpoint': None,
    'model_path': None
}
```
```

*Lưu ý về các chế độ suy luận:* Chế độ `greedy`, `sampling`, `aug_sample` được điều khiển bằng cách bật/tắt trong file `configs/*.py` (ví dụ `PPO/configs/fjsp.py:67-71`):
```python
'test': {
    'inference_type': [
        'greedy',
        # 'sampling',
        # 'aug_sample',
    ],
```

---

## 10. Cách Đánh Giá và Tính Toán Gap

### 10.1. Tệp đối chứng trong `or_solution`
- **Vị trí mã nguồn:** `PPO/utils.py:230-233` và `PPO/configs/fjsp.py:284-293`.
- **Quy tắc đường dẫn:**
  ```python
  d_path = Path(path)
  source, data_name = d_path.parent.name, d_path.name
  file_path = f'../data/FJSP/TNNLS/or_solution/{source}/solution_{data_name}.npy'
  ```
  - Ví dụ: Dataset `data/FJSP/TNNLS/SD1/10x5` sẽ so với file `data/FJSP/TNNLS/or_solution/SD1/solution_10x5.npy`.
  - Dataset `data/FJSP/TNNLS/SD2/10x5+mix` sẽ so với file `data/FJSP/TNNLS/or_solution/SD2/solution_10x5+mix.npy`.
  - Cột đầu tiên `solution[:, 0]` chính là giá trị makespan do OR-Tools giải được (`PPO/configs/fjsp.py:290`).
  *(Đối với các benchmark chuẩn như Brandimarte hay Hurink, baseline nạp từ `data/FJSP/TNNLS/BenchData/BenchDataSolution.csv` cột `ub` theo `PPO/configs/fjsp.py:264-267`).*

### 10.2. Công thức tính Gap
- **Vị trí mã nguồn:** `PPO/SchedulingEvaluator.py:56-57`:
  ```python
  gap = (score_lists / makespan - 1) * 100
  gap = np.mean(gap)
  ```
- **Công thức toán học:**
  $$\text{Gap (\%)} = \frac{1}{K} \sum_{i=1}^{K} \left( \frac{\text{Makespan}_{\text{model}}^{(i)}}{\text{Makespan}_{\text{baseline}}^{(i)}} - 1 \right) \times 100\%$$

---

## 11. Cấu Hình Môi Trường Hiện Tại

- **Card đồ họa (GPU):**
  - Thiết bị: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**
  - Bộ nhớ VRAM: **6144 MiB (6 GB)**
  - NVIDIA Driver: **581.32**
  - CUDA Driver Hỗ trợ: **13.0**
- **Môi trường Conda có sẵn PyTorch GPU (`ec_nco`):**
  - Đường dẫn: `D:\Miniconda\miniconda3\envs\ec_nco\python.exe`
  - Python: **3.10.19** (64-bit AMD64)
  - PyTorch: **2.7.1+cu118**
  - `torch.cuda.is_available()`: **True**
- **Phiên bản yêu cầu trong `README.md:164-168`:**
  - `torch == 2.3.1 (CUDA 12.1)`
  - `ortools == 9.11.4210`
  - `numpy`, `pandas`
