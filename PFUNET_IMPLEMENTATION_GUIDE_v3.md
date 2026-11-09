# PF-UNet trên codebase ReSched: hướng dẫn triển khai chi tiết (v3)

> Phiên bản này thay thế v1 và v2. Khác biệt chính: **ReSched giữ nguyên làm baseline trong `PPO/`**, PF-UNet được **thêm vào** (package riêng cùng một hook chọn mô hình), không sao chép cây thư mục. Nó dựa trên báo cáo `repo_facts.md` (đã xác minh bằng cách đọc code, có `tệp:dòng`).
> Mã trong tài liệu là **mã giả, chưa chạy thử**. Ưu tiên đúng ngữ nghĩa và hình dạng tensor hơn là cú pháp. Nếu mã giả mâu thuẫn với code thực tế, **dừng lại và báo cáo**, đừng tự đoán.
> Khi tài liệu này mâu thuẫn với đề cương PF-UNet, làm theo tài liệu này và ghi vào `PPO/pfunet/docs/design_changes.md`.

---

## 0. Cách dùng tài liệu

1. Đọc mục 2 (sự thật đã xác minh) và mục 3 (quyết định thiết kế đã chốt).
2. Làm theo thứ tự giai đoạn ở mục 7. Mỗi giai đoạn có tiêu chí đạt; không đạt thì dừng.
3. Mã giả ở mục 5 là bản đặc tả để cài đặt; các bài kiểm thử ở mục 5.9 là bắt buộc.
4. Máy hiện tại là RTX 3050 6 GB: chỉ dùng để kiểm tra **chạy được và chạy đúng**. Xem mục 6.

---

## 1. Mục tiêu và phạm vi

**Mục tiêu:** cài PF-UNet làm chính sách mới cho FJSP trong khung huấn luyện/đánh giá của ReSched, rồi so với ReSched.

**Ý tưởng:** mỗi bước, đưa các ứng viên (operation head của mỗi job) lên một canvas chuẩn hóa cho từng máy (x = thời điểm bắt đầu hiệu dụng, y = công việc còn lại của job). Một U-Net dùng chung trọng số giữa các máy xử lý canvas; điểm của từng cặp (operation, máy) được đọc ngược bằng lấy mẫu song tuyến tính.

**Giữ nguyên (không sửa logic):** môi trường, reward, bộ sinh dữ liệu, CP-SAT, `PPOTrainer.py`, `SchedulingEvaluator.py`, dữ liệu, và toàn bộ mô hình ReSched (`SchedulingModel.py`). ReSched vẫn là **baseline**, chạy bằng chính code gốc.

**Thêm vào:** package `PPO/pfunet/` (mô hình PF-UNet, bộ xây dựng quan sát, bộ mô phỏng tham chiếu, kiểm thử) và một **hook chọn mô hình** (vài dòng ở `SchedulingRunner.py` và một khối cấu hình ở `configs/fjsp.py`, mặc định vẫn là ReSched).

**Giao thức dùng chung:** huấn luyện (PPO, ngân sách, siêu tham số), kiểm thử (greedy, sampling, `aug_sample`), dữ liệu và cách tính gap của ReSched áp dụng nguyên cho cả hai mô hình. Khác biệt duy nhất giữa hai lần chạy là `model_type`.

**Ngoài phạm vi:** sửa logic môi trường hay reward; tối ưu tốc độ trước P7; mọi thứ không có ở đây.

---

## 2. Sự thật đã xác minh và hệ quả cho PF-UNet

| Sự thật (`repo_facts.md`) | Hệ quả khi cài PF-UNet |
|---|---|
| Mô hình nhận **tuple 8 tensor** `(min_duration, duration, o_at, m_at, dependency_o, dependency_m, action_mask, rev_pos)` do `build_model_input` (`SchedulingModel.py:8-45`) tạo ra; không có job id | PF-UNet **giữ nguyên tuple 8 phần tử** (để ReSched không bị ảnh hưởng) và suy nhóm job từ `dependency_o` (mục 3, D2). Môi trường có `state.job_idx` `(B, N_rem)` (`SchedulingEnvironment.py:150, 301`); chỉ dùng nó trong bài kiểm thử để xác nhận cách suy nhóm |
| Đầu ra PPO: `[seq_id, machine_id], prob_selected, value.squeeze(), prob`; `prob` là xác suất `(B, N_rem, M)` sau softmax, ô bị che bằng 0 (`SchedulingModel.py:112`) | PF-UNet phải trả đúng 4 phần tử này, với `prob` ở **không gian của môi trường** `(B, N_rem, M)` |
| Trainer/Evaluator gọi: `model(state)`, `model(state, action=action)`, `set_decode_type('greedy' \| 'sampling' \| 'teacher_forcing')`, `train()`, `eval()`, `parameters()`, `state_dict()`, `load_state_dict()`; `Model(**model_params)` ở `Runner:44`; `model_old` là bản sao dùng khi rollout | Giữ tên lớp, tên file, chữ ký và các phương thức đó |
| `duration (B,N_rem,M)`, `min_duration (B,N_rem)`, `m_AT (B,M)`, `o_AT (B,N_rem)`, `position_rev (B,N_rem) int32`, `action_mask (B,N_rem,M) bool`, `o2o_mask (B,N_rem,N_rem) bool`, `o2m_mask (B,N_rem,M) bool` | Hình dạng đầu vào cho bộ xây dựng quan sát |
| Máy không tương thích có `duration = 0`; `action_mask` đã loại các ô này | Mặt nạ tương thích lấy từ `action_mask` (head) và `dependency_m` (mọi operation), **không** suy từ `duration == 0` ở nơi khác |
| `o_AT` của operation không phải head giữ giá trị 0; mô hình chỉ dùng `o_AT` ở head (`SchedulingModel.py:74-76`) | Chỉ đọc `o_at` tại các hàng head |
| Head = hàng có `action_mask[b,o,:].any()`; mỗi job có tối đa một head | `head = act_mask.any(-1)`; mỗi head là một slot job (mục 3, D4) |
| Môi trường **không padding**; mỗi batch chỉ chứa instance cùng cỡ `(N, M)`; cấu trúc số operation mỗi job sinh một lần rồi lặp cho cả batch; mọi instance trong batch **kết thúc cùng bước** | `M` cố định trong batch, **không cần padding máy**. Số head mỗi instance có thể khác nhau (job hoàn tất ở các bước khác nhau), nên các slot job được đệm đến số head lớn nhất của batch và che bằng `has_head`. Không cần xử lý instance kết thúc lệch bước |
| Bước cuối tự động (`SchedulingEnvironment.py:98-105`), không tính vào reward; mô hình chỉ quyết định khi `N_rem ≥ 2` | Mô hình không bao giờ thấy `N_rem = 1` |
| Reward do môi trường tính; thời gian đã chia `scaler`, gốc là min cục bộ | Làm việc trong **khung của môi trường**; S_t triệt tiêu `scaler`; không khôi phục thời gian tuyệt đối |
| Gap = trung bình theo instance của `(makespan_model / makespan_baseline − 1) × 100` (`SchedulingEvaluator.py:56-57`); baseline lấy từ `or_solution/{SD1\|SD2}/solution_<name>.npy` cột 0; benchmark dùng cột `ub` của `BenchDataSolution.csv` | Báo cáo gap đúng công thức này (trung bình của tỉ số theo instance, không phải tỉ số của các trung bình) |
| Chế độ suy luận bật/tắt trong `configs/*.py` (`'test': {'inference_type': [...]}`) | Dùng đúng cơ chế này |
| SD1 sinh được cỡ tùy ý (`SD1FJSPGenerator.py`); không có generator riêng cho SD2 (dữ liệu SD2 có sẵn dạng tệp); `FlexibleJobShopProblemGenerator` (`SchedulingGenerator.py:126-152`) có `min_operation`, `max_operation` | Cỡ lớn 100×10, 200×5: dùng SD1 generator; SD2 cỡ lớn cần cấu hình `FlexibleJobShopProblemGenerator` (kiểm tra ở P1) |
| Đường dẫn dữ liệu là tương đối `../data/...` và `../ckpt/...` | PF-UNet nằm trong `PPO/pfunet/` và chạy từ `PPO/` như baseline, nên các đường dẫn này đúng sẵn |
| Môi trường Python: Conda `ec_nco`, Python 3.10.19, PyTorch 2.7.1+cu118; README yêu cầu torch 2.3.1 | Dùng môi trường hiện có. Từ PyTorch 2.6, `torch.load` mặc định `weights_only=True`; nếu nạp checkpoint cũ lỗi, thử `weights_only=False` (chỉ cho checkpoint tin cậy của repo) |
| GPU: RTX 3050 Laptop 6 GB | Xem mục 6 |

---

## 3. Quyết định thiết kế đã chốt (khác đề cương)

**D1. Thêm, không thay.** ReSched giữ nguyên làm baseline. PF-UNet là package mới `PPO/pfunet/`; lớp `SchedulingModel` của nó có **cùng chữ ký** với lớp gốc (`__init__(**model_params)`, `forward(state, action=None)`, `set_decode_type`, `train`, `eval`, `state_dict`). Hook chọn mô hình là thay đổi duy nhất trong code gốc (mục 5.2). Trainer, Evaluator và môi trường chạy nguyên vẹn.

**D2. Suy nhóm job từ `dependency_o`, không đổi tuple.** Tuple 8 phần tử giữ nguyên. Với mỗi operation head `h`, hàng `dependency_o[b, h, :]` được kỳ vọng đánh dấu `h` và các operation kế tiếp của cùng job (cạnh O2O hướng về sau cùng hop connection tới mọi operation đứng sau, theo bài báo ReSched), tức là toàn bộ phần còn lại của job. Từ đó `W_j` (tổng `min_duration` còn lại) và `n_j` (số operation còn lại) tính bằng một phép nhân ma trận, không cần job id.
**Phải kiểm chứng bằng `test_dependency_direction` ở P3:** với mọi head `h`, `dependency_o[b, h, o] == (state.job_idx[b, o] == state.job_idx[b, h])` (môi trường có `state.job_idx`, chỉ dùng trong test, không dùng trong mô hình). Nếu ma trận bị chuyển vị thì dùng `dependency_o.transpose(1, 2)`. Nếu cả hai đều sai (ma trận không mang cấu trúc job thuần), **dừng và báo cáo**; phương án dự phòng là truyền thêm `job_idx` qua một hàm `build_model_input` riêng cho PF-UNet, khi đó phải xem lại cách Trainer lưu trạng thái.

**D3. Đặc trưng tiến độ.** Mô hình không nhận được số operation ban đầu `N_j0`. Thay `n_j / N_j0` bằng `n_j / max_j n_j` (kênh 5 và đặc trưng thô thứ 7). Ghi vào `design_changes.md`.

**D4. Slot job = các head.** `M` cố định trong batch nên không cần padding máy. Mỗi instance có `J_a` head (số job còn việc). Mô hình dùng `J = max_b J_a` slot, xếp các head theo thứ tự chỉ số operation tăng dần và đệm phần còn lại; `has_head` che các slot đệm. Thứ tự slot không có ý nghĩa vì mô hình bất biến theo hoán vị job.

**D5. Chọn hành động.** Mô hình tính logit trên `[B, M, J]`, đổi sang xác suất `prob` ở không gian môi trường `[B, N_rem, M]` bằng `scatter_add_` (không dùng `scatter_`, vì các slot đệm không có head vẫn trỏ tới một hàng operation thật và sẽ ghi 0 đè lên nó). Sau đó **sao chép nguyên văn** khối chọn hành động của mô hình gốc (greedy, sampling, teacher forcing) cho phần còn lại của `forward`.

**D6. Critic.** `value` = MLP của trung bình theo máy của `g` (vector bottleneck sau attention giữa các máy).

---

## 4. Khởi tạo thư mục

```bat
cd D:\ReSched
git checkout -b pfunet
git tag baseline
mkdir PPO\pfunet
mkdir PPO\pfunet\tests
mkdir PPO\pfunet\docs
```

- **Không sao chép** `PPO/`. ReSched ở nguyên vị trí và là baseline; PF-UNet là package con `PPO/pfunet/`.
- Chạy mọi lệnh từ `D:\ReSched\PPO` như README hướng dẫn (các đường dẫn `../data`, `../ckpt` đúng sẵn). Chuyển mô hình bằng khóa `model_type` trong cấu hình.
- Chỉ **hai** tệp gốc được phép sửa, ở mức hook (mục 5.2): `PPO/SchedulingRunner.py` và `PPO/configs/fjsp.py`. Mọi thứ khác là tệp mới.
- Trước khi sửa, ghi lại kết quả greedy của ReSched trên 20 instance cố định. Sau khi thêm hook, `test_baseline_unchanged` (mục 5.9) phải cho kết quả **trùng khớp từng bit**.

Cấu trúc cuối cùng:

```text
PPO/
  SchedulingRunner.py         # SỬA: hook chọn mô hình (vài dòng)
  configs/fjsp.py             # SỬA: khóa model_type và PFUNET_PARAMS (mặc định vẫn là ReSched)
  pfunet/                     # MỚI
    __init__.py
    modules.py                # Obs, build_pf_obs, rasterize, readout, UNet2D, MachineAttention
    model.py                  # SchedulingModel (PF-UNet), cùng chữ ký với mô hình gốc
    ref_sim.py                # mô phỏng tham chiếu bằng Python thuần
    docs/repo_facts.md        # kết quả P0 (đã có, chuyển vào đây)
    docs/design_changes.md    # nhật ký thay đổi so với đề cương
    tests/
      test_baseline_unchanged.py
      test_ref_parity.py
      test_dependency_direction.py
      test_obs_builder.py
      test_rasterize.py
      test_invariance.py
      test_policy_interface.py
      test_overfit_one.py
```

---

## 5. Mã giả chi tiết

### 5.1 Sơ đồ luồng dữ liệu

```text
env.state ──build_model_input──► 8-tuple ──build_pf_obs──► Obs
                                                           │
   Obs ──rasterize_obs──► canvas [B*M, 8, H, W]
       ──UNet.encode──► skips, h [B*M, 256, H/8, W/8]
       ──pool theo máy──► g [B, M, 256] ──MachineAttention──► g'
       ──FiLM(g') áp lên h──► h'
       ──UNet.decode(h', skips)──► d0 [B*M,32,H,W], d1 [B*M,64,H/2,W/2]
       ──readout tại (x,y) + raw + g'──► feat [B,M,J,359]
       ──MLP──► logits [B,M,J] ──(softmax, scatter_add_)──► prob [B,N_rem,M]
                                                           │
   prob ──[khối chọn hành động gốc]──► ([seq_id, machine_id], prob_selected, value, prob)
```

### 5.2 Hook chọn mô hình (thay đổi duy nhất trong code gốc)

```python
# PPO/configs/fjsp.py: thêm khóa model_type. MẶC ĐỊNH vẫn là ReSched.
model_params = {'model_type': 'resched', ...}      # các khóa gốc giữ nguyên
# Để chạy PF-UNet: model_params = PFUNET_PARAMS (khối ở mục 6, có 'model_type': 'pfunet')

# PPO/SchedulingRunner.py, quanh dòng 44 (nơi gọi Model(**model_params))
from pfunet.model import SchedulingModel as PFUNetModel          # THÊM (đầu tệp)
...
mp = dict(model_params)
model_type = mp.pop('model_type', 'resched')                     # ReSched không nhận khóa này
ModelCls = PFUNetModel if model_type == 'pfunet' else Model
self.model = ModelCls(**mp)                                      # thay cho Model(**model_params)
```

Ghi chú:

- Tìm nơi `model_old` được tạo (Runner hay Trainer). Hook phải áp dụng cho **cả hai** bản sao, nếu không rollout và cập nhật sẽ dùng hai kiến trúc khác nhau.
- Nếu cấu hình được truyền vào Runner theo cách khác (không qua từ điển `model_params`), đặt khóa `model_type` ở chỗ tương ứng; giữ nguyên nguyên tắc: **mặc định là ReSched** và hook chỉ thêm một nhánh.
- Sau khi thêm hook, chạy `test_baseline_unchanged` trước mọi việc khác.

### 5.3 `build_pf_obs` (vector hóa hoàn toàn)

```python
Obs = namedtuple('Obs', 'xy feats raw mask_pair const head_idx has_head Ja S')

def build_pf_obs(state, eps=1e-6):
    (min_dur, dur, o_at, m_at, dep_o, dep_m, act_mask, rev_pos) = state    # tuple 8 phần tử, KHÔNG đổi
    B, N, M = dur.shape
    dev = dur.device

    # (1) slot job = các operation head, xếp theo chỉ số operation tăng dần; phần còn lại là đệm
    head = act_mask.any(-1)                                                 # [B, N]
    J = int(head.sum(1).max().item())                                       # số slot của batch
    order = torch.argsort((~head).to(torch.int8), dim=1, stable=True)[:, :J]   # [B, J]
    has_head = head.gather(1, order)                                        # [B, J]
    safe = order                                                            # chỉ số operation hợp lệ, kể cả slot đệm

    # (2) phần còn lại của job = hàng của head trong dependency_o (D2; cần test_dependency_direction)
    G = dep_o.gather(1, order[..., None].expand(B, J, N)) & has_head[..., None]   # [B, J, N] bool
    Gf = G.to(dur.dtype)
    W = (Gf * min_dur[:, None, :]).sum(-1)                                  # [B, J] tổng min_duration còn lại (gồm head)
    n = Gf.sum(-1)                                                          # [B, J] số operation còn lại

    # (3) dữ liệu của operation head
    R     = o_at.gather(1, safe)                                            # [B, J]  thời điểm sẵn sàng của job
    p_min = min_dur.gather(1, safe)                                         # [B, J]
    p_hd  = dur.gather(1, safe[..., None].expand(B, J, M))                  # [B, J, M]
    ok_hd = act_mask.gather(1, safe[..., None].expand(B, J, M)) & has_head[..., None]   # [B, J, M]
    nM_hd = ok_hd.sum(-1).clamp_min(1).to(dur.dtype)                        # [B, J] số máy tương thích của head

    # (4) chuẩn hóa (định nghĩa chính thức, trong khung của môi trường)
    A  = m_at                                                               # [B, M]
    t0 = A.min(1).values                                                    # [B]
    Rt = torch.maximum(R, t0[:, None])
    LBj = (Rt + W).masked_fill(~has_head, float('-inf')).max(1).values
    LB = torch.maximum(LBj, A.max(1).values)
    S  = (LB - t0).clamp_min(eps)                                           # [B]

    # (5) bố cục [B, M, J]
    P  = p_hd.permute(0, 2, 1)                                              # [B, M, J]
    OK = ok_hd.permute(0, 2, 1)                                             # [B, M, J] bool
    ST = torch.maximum(R[:, None, :], A[:, :, None])                        # [B, M, J]
    S3 = S[:, None, None]
    x  = ((ST - t0[:, None, None]) / S3).clamp(0, 1)
    y  = (W / S[:, None])[:, None, :].expand(B, M, J).clamp(0, 1)
    wait = (ST - A[:, :, None]) / S3
    prog = (n / n.max(1, keepdim=True).values.clamp_min(1))[:, None, :].expand(B, M, J)   # D3
    flex = (nM_hd / M)[:, None, :].expand(B, M, J)
    pmin_ratio = P / p_min.clamp_min(eps)[:, None, :]

    xy    = torch.stack([x, y], -1)                                         # [B, M, J, 2]
    feats = torch.stack([torch.ones_like(x), P / S3, flex, wait, prog], -1) # [B, M, J, 5]  kênh 1..5
    raw   = torch.stack([P / S3, pmin_ratio, wait, x, y, flex, prog], -1)   # [B, M, J, 7]

    # (6) kênh hằng của máy: 6 (máy rảnh), 7 (tỉ lệ tải tương lai)
    nM_all = dep_m.sum(-1).clamp_min(1).to(dur.dtype)                       # [B, N]
    L  = ((dur / nM_all[..., None]) * (~head)[..., None]).sum(1)            # [B, M]  tải của operation chưa tới lượt
    c6 = (A - t0[:, None]) / S[:, None]
    c7 = L / (L.sum(1, keepdim=True) + eps)
    const = torch.stack([c6, c7], -1)                                       # [B, M, 2]

    Ja = has_head.sum(1).clamp_min(1).to(dur.dtype)                         # [B]
    return Obs(xy, feats, raw, OK, const, order, has_head, Ja, S)
```

### 5.4 `rasterize` (splat song tuyến tính)

```python
def rasterize(xy, vals, mask, H, W):
    # xy [B,M,J,2] trong [0,1]; vals [B,M,J,C]; mask [B,M,J] bool
    B, M, J, C = vals.shape
    gx, gy = xy[..., 0] * (W - 1), xy[..., 1] * (H - 1)
    x0 = gx.floor().clamp(0, W - 1); y0 = gy.floor().clamp(0, H - 1)
    fx, fy = gx - x0, gy - y0
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0 + 1).clamp(max=W - 1), (y0 + 1).clamp(max=H - 1)
    canvas = vals.new_zeros(B * M, H * W, C)
    for xi, yi, w in ((x0, y0, (1 - fx) * (1 - fy)), (x1, y0, fx * (1 - fy)),
                      (x0, y1, (1 - fx) * fy),       (x1, y1, fx * fy)):
        idx = (yi * W + xi).reshape(B * M, J)
        contrib = (vals * (w * mask)[..., None]).reshape(B * M, J, C)
        canvas.scatter_add_(1, idx[..., None].expand(-1, -1, C), contrib)
    return canvas.view(B * M, H, W, C).permute(0, 3, 1, 2)         # [B*M, C, H, W]

def rasterize_obs(obs, H, W):
    B, M, J = obs.mask_pair.shape
    c = rasterize(obs.xy, obs.feats, obs.mask_pair, H, W)           # [B*M, 5, H, W]
    c = c / obs.Ja.repeat_interleave(M)[:, None, None, None]        # chia J_a (kênh 1..5)
    const = obs.const.reshape(B * M, 2, 1, 1).expand(-1, -1, H, W)  # kênh 6, 7
    ctx = c[:, :1].view(B, M, 1, H, W).mean(1, keepdim=True)        # kênh 8: trung bình qua máy của kênh 1
    ctx = ctx.expand(B, M, 1, H, W).reshape(B * M, 1, H, W)
    return torch.cat([c, const, ctx], 1)                            # [B*M, 8, H, W]
```

Khối lượng được bảo toàn: tổng kênh 1 trên canvas (trước khi chia `J_a`) bằng số ứng viên khả thi của máy đó, vì trọng số bilinear của mỗi ứng viên cộng lại bằng 1, kể cả khi chỉ số bị clamp.

### 5.5 `readout` (đọc ngược)

```python
def readout(maps, xy, B, M):
    # maps [B*M, C, h, w] (mức đầy đủ hoặc một nửa); xy [B, M, J, 2] trong [0,1]
    J = xy.shape[2]
    grid = xy.reshape(B * M, J, 1, 2) * 2 - 1                       # thứ tự (x, y) = (rộng, cao)
    out = F.grid_sample(maps, grid, mode='bilinear',
                        padding_mode='border', align_corners=True)  # [B*M, C, J, 1]
    return out.squeeze(-1).permute(0, 2, 1).reshape(B, M, J, -1)    # [B, M, J, C]
```

Quy ước nhất quán với splat: `gx = x*(W-1)`, `gy = y*(H-1)`, `align_corners=True`, `grid[...,0] = x`, `grid[...,1] = y`.

### 5.6 `UNet2D` và `MachineAttention`

```python
class ConvBlock(nn.Module):                      # Conv3x3 + GroupNorm + GELU
    def __init__(self, c_in, c_out, groups=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                                 nn.GroupNorm(groups, c_out), nn.GELU())
    def forward(self, x): return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, c_in, chs=(32, 64, 128, 256), blocks=2, groups=8):
        super().__init__()
        L = len(chs)
        self.enc = nn.ModuleList([nn.Sequential(*[ConvBlock(c_in if (i == 0 and b == 0) else
                                                           (chs[i] if b > 0 else chs[i-1]), chs[i], groups)
                                                  for b in range(blocks)]) for i in range(L)])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i], 3, stride=2, padding=1) for i in range(L - 1)])
        self.up   = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                 nn.Conv2d(chs[i + 1], chs[i], 3, padding=1)) for i in range(L - 1)])
        self.dec  = nn.ModuleList([nn.Sequential(*[ConvBlock(2 * chs[i] if b == 0 else chs[i], chs[i], groups)
                                                   for b in range(blocks)]) for i in range(L - 1)])
    def encode(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x); x = self.down[i](x)
        return skips, x                                              # x: bottleneck
    def decode(self, h, skips):
        outs = []
        for i in reversed(range(len(self.dec))):                     # mức sâu -> mức nông
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], 1))
            outs.append(h)
        return outs[::-1]                                            # [d0 (đầy đủ, 32), d1 (1/2, 64), ...]

class MachineAttention(nn.Module):               # self-attention giữa các máy, không có mã hóa vị trí
    def __init__(self, dim=256, heads=4, layers=2, ff=512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, ff, batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
    def forward(self, g): return self.enc(g)                         # [B, M, dim]
```

Không có mã hóa vị trí giữa các máy nên mô hình tương đương theo hoán vị máy. `H` và `W` phải chia hết cho `2^(L-1) = 8`.

### 5.7 `SchedulingModel` (PF-UNet, trong `PPO/pfunet/model.py`)

```python
class SchedulingModel(nn.Module):                 # cùng tên và chữ ký với lớp gốc (D1)
    def __init__(self, **p):
        super().__init__()
        self.H, self.W = p['canvas_hw']
        D = p['unet_channels'][-1]
        self.unet = UNet2D(p['c_in'], p['unet_channels'], p['blocks_per_level'], p['gn_groups'])
        self.mach_attn = MachineAttention(D, p['attn_heads'], p['attn_layers'], p['attn_ff'])
        self.film = nn.Linear(D, 2 * D)
        rd = p['unet_channels'][0] + p['unet_channels'][1] + p['raw_dim'] + D      # 32 + 64 + 7 + 256 = 359
        self.head = MLP([rd, *p['head_hidden'], 1])                                  # [128, 64] -> 1
        self.critic = MLP([D, p['critic_hidden'], 1])
        self.decode_type = None
    def set_decode_type(self, t): self.decode_type = t

    def core(self, obs):
        B, M, J = obs.mask_pair.shape
        canvas = rasterize_obs(obs, self.H, self.W)                                  # [B*M, 8, H, W]
        skips, h = self.unet.encode(canvas)
        g = self.mach_attn(h.mean((2, 3)).view(B, M, -1))                            # [B, M, D]
        gam, bet = self.film(g).chunk(2, -1)
        h = h * (1 + gam.reshape(B * M, -1, 1, 1)) + bet.reshape(B * M, -1, 1, 1)
        d0, d1 = self.unet.decode(h, skips)[:2]
        f0, f1 = readout(d0, obs.xy, B, M), readout(d1, obs.xy, B, M)
        gk = g[:, :, None, :].expand(B, M, J, -1)
        score = self.head(torch.cat([f0, f1, obs.raw, gk], -1)).squeeze(-1)          # [B, M, J]
        logits = score.masked_fill(~obs.mask_pair, float('-inf'))
        value = self.critic(g.mean(1)).squeeze(-1)                                   # [B]
        return logits, value

    def forward(self, state, action=None):
        (min_dur, dur, o_at, m_at, dep_o, dep_m, act_mask, rev_pos) = state    # tuple 8 phần tử như ReSched
        B, N, M = dur.shape
        obs = build_pf_obs(state)
        logits, value = self.core(obs)                                               # [B,M,J], [B]
        J = logits.shape[-1]

        # logit [B,M,J] -> xác suất trong không gian môi trường [B,N_rem,M]
        flat = logits.permute(0, 2, 1).reshape(B, J * M)                             # thứ tự (j, k)
        p_jm = torch.softmax(flat, -1).view(B, J, M) * obs.has_head[..., None]
        prob = dur.new_zeros(B, N, M)
        prob.scatter_add_(1, obs.head_idx.clamp_min(0)[..., None].expand(B, J, M), p_jm)

        # ===== từ đây: SAO CHÉP NGUYÊN VĂN khối chọn hành động của SchedulingModel gốc =====
        # (PPO/SchedulingModel.py:~95-112: greedy / sampling / teacher_forcing dựa trên `prob`,
        #  rồi `return [seq_id, machine_id], prob[arange, seq_id, machine_id], value.squeeze(), prob`)
```

Ghi chú:

- Nếu khối chọn hành động gốc làm việc với `prob` đã flatten `(B, N_rem * M)`, giữ đúng cách flatten và giải mã `seq_id`, `machine_id` của bản gốc.
- Ở chế độ `teacher_forcing`, lấy `prob[arange(B), seq, machine]` từ `prob` đã scatter (nhất quán với mọi chế độ khác).
- Không bao giờ để `prob` chứa NaN: mỗi instance có ít nhất một cặp khả thi vì `N_rem ≥ 2`.

### 5.8 `ref_sim.py` (mô phỏng tham chiếu)

```python
class RefSim:
    def __init__(self, dur, jobs):            # dur[o, k] (inf nếu không tương thích); jobs = [[op ids theo thứ tự], ...]
        self.dur, self.jobs = dur, jobs
        self.R = np.zeros(len(jobs)); self.A = np.zeros(dur.shape[1])
        self.ptr = [0] * len(jobs)             # operation kế tiếp của mỗi job
    def heads(self):
        return [(j, ops[self.ptr[j]]) for j, ops in enumerate(self.jobs) if self.ptr[j] < len(ops)]
    def step(self, j, k):                      # chọn head của job j, máy k
        o = self.jobs[j][self.ptr[j]]
        start = max(self.R[j], self.A[k]); finish = start + self.dur[o, k]
        self.R[j] = finish; self.A[k] = finish; self.ptr[j] += 1
        return finish
    def makespan(self): return self.R.max()

def ref_obs(sim):                              # tính Obs bằng vòng lặp thuần theo công thức ở mục 6.5 của đề cương
    ...                                        # dùng đơn vị tuyệt đối; kết quả x, y, tỉ lệ đặc trưng không phụ thuộc khung
```

### 5.9 Kiểm thử (bắt buộc)

```python
# test_baseline_unchanged.py: hồi quy; chạy SAU khi thêm hook, so với kết quả ghi lại TRƯỚC khi sửa (git tag baseline)
set_all_seeds(0)
res_after = evaluate(model_type='resched', ckpt='../ckpt/PPO/SD1-10x05.pth',
                     inference='greedy', instances=fixed_20)
assert res_after.makespans == res_before.makespans              # greedy là tất định: phải trùng khớp từng bit

# test_dependency_direction.py: kiểm chứng D2 trên môi trường thật (state.job_idx chỉ dùng trong test)
for state in reachable_states(200 rollouts ngẫu nhiên, SD1 và SD2):
    head = act_mask.any(-1)                                      # [B, N]
    same_job = job_idx[:, :, None] == job_idx[:, None, :]        # [B, N, N]
    assert (dependency_o[head] == same_job[head]).all()          # hàng của head = phần còn lại của job
    # nếu sai: thử dependency_o.transpose(1, 2); nếu vẫn sai: DỪNG và báo cáo (D2)

# test_ref_parity.py: môi trường ReSched so với RefSim
for trial in range(1000):
    env.reset(small_instance); sim = RefSim(*to_ref(env))
    while not env.done:
        (seq, mach) = random_valid_action(env.state.action_mask)        # theo action_mask
        j = env.state.job_idx[0, seq]
        env.step((seq, mach)); sim.step(j, mach)
    assert abs(env_makespan(env) - sim.makespan()) < 1e-6               # tìm cách lấy makespan từ env ở P1

# test_obs_builder.py: Obs vector hóa so với ref_obs từ RefSim, 200 trạng thái ngẫu nhiên SD1 và SD2 (10x5, 20x10)
assert torch.allclose(obs.xy, ref.xy, atol=1e-5) and torch.allclose(obs.feats, ref.feats, atol=1e-5)
assert (obs.xy >= 0).all() and (obs.xy <= 1).all()
assert (obs.has_head.sum(1) == n_alive_jobs_from_env_job_idx).all()     # số head = số job còn việc (đếm từ env.state.job_idx)

# test_rasterize.py
assert rasterize(one_point_at_pixel_center)  is one-hot
assert rasterize(one_point_at_midpoint)      has 0.25 ở 4 pixel kề
assert canvas[:, 0].sum((1, 2)) (trước chia Ja) == mask_pair.sum(-1) (số ứng viên mỗi máy)
assert readout(constant_map, xy) == constant

# test_invariance.py (trên core(obs), không qua môi trường)
core(perm_J(obs)) == perm_J(core(obs))            # hoán vị job: logits hoán vị theo, atol 1e-5
core(perm_M(obs)) == perm_M(core(obs))            # hoán vị máy
core(scale(state, c)) == core(state)              # nhân dur, o_at, m_at với c > 0: logits không đổi

# test_policy_interface.py: cùng một state, so với mô hình gốc
for mode in ('greedy', 'sampling', 'teacher_forcing'):
    out = model(state) hoặc model(state, action=...)
    assert len(out) == 4 and out[1].shape == (B,) and out[2].shape == (B,) and out[3].shape == (B, N, M)
    assert torch.allclose(out[3].flatten(1).sum(-1), ones)             # tổng xác suất bằng 1
    assert (out[3][~action_mask] == 0).all()                           # ô không khả thi bằng 0
loss = out[2].sum() + out[1].log().sum(); loss.backward()             # gradient chảy qua mọi tham số
```

---

## 6. Cấu hình và hồ sơ chạy

**Nguyên tắc:** kiến trúc **không đổi** giữa hồ sơ `smoke` (máy hiện tại) và `full` (máy mạnh). Chỉ đổi cỡ batch, số update, số instance và kích thước chunk khi đánh giá.

Khối `PFUNET_PARAMS` trong `PPO/configs/fjsp.py` (dùng khi `model_type = 'pfunet'`; các khóa của ReSched giữ nguyên):

```python
PFUNET_PARAMS = {
    'model_type': 'pfunet',
    'canvas_hw': (64, 64), 'c_in': 8,
    'unet_channels': (32, 64, 128, 256), 'blocks_per_level': 2, 'gn_groups': 8,
    'attn_layers': 2, 'attn_heads': 4, 'attn_ff': 512,
    'head_hidden': (128, 64), 'critic_hidden': 128, 'raw_dim': 7,
}
```

| Tham số | `smoke` (RTX 3050 6 GB) | `full` |
|---|---|---|
| Cỡ huấn luyện | SD1 10×5 | SD1/SD2 10×5, rồi 20×10 |
| Batch huấn luyện B | 4 | 50 |
| Số update | 20 đến 50 | 400 (nhỏ), 10000 (lớn) |
| Mini-batch PPO | 2 | 10 |
| Số instance validation/test | 10 | 100 |
| Suy luận | greedy; `aug_sample` trên 2 instance | greedy, sampling, `aug_sample` |
| `eval_chunk` (tách `B×100` khi `aug_sample`) | 100 (hạ xuống nếu OOM) | theo VRAM |
| Số seed | 1 | ít nhất 4 |

Điều chỉnh khi hết bộ nhớ, **theo thứ tự**: giảm `B` và mini-batch, giảm `eval_chunk`, giảm số instance. **Không** thu nhỏ mô hình để chạy trên máy này (kể cả `canvas_hw`, `unet_channels`).

Khi chạy `smoke`, ghi lại cho mỗi cấu hình: VRAM đỉnh (`torch.cuda.max_memory_allocated`), thời gian mỗi update, thời gian mỗi instance khi suy luận. Số liệu này để ước lượng khi chuyển sang máy mạnh.

Siêu tham số PPO (theo ReSched/DANIEL, Phụ lục E): `gamma = 1.0`, `lambda = 0.98`, `clip = 0.2`, `vf_coef = 0.5`, `ent_coef = 0.01`, `K = 4`. Tốc độ học khởi đầu 5e-5 (quét {2e-5, 5e-5, 1e-4} ở máy mạnh). Tìm tên khóa tương ứng trong `configs/fjsp.py` và ghi lại ánh xạ vào `design_changes.md`.

---

## 7. Kế hoạch theo giai đoạn và tiêu chí đạt

Chỉ sang giai đoạn sau khi giai đoạn trước đạt. Không đạt thì **dừng và báo cáo**, không chỉnh ngưỡng.

**P0 (đã xong): xác minh repo.** Đã có `repo_facts.md`. Còn ba điểm mở cần xác minh ở P1:
(a) nơi tạo `model_old` (Runner hay Trainer), để hook chọn mô hình áp dụng cho cả hai bản sao;
(b) cách lấy makespan và lịch từ môi trường để so với `RefSim`;
(c) `torch.load` checkpoint hoạt động trên PyTorch 2.7.1 (nếu không, `weights_only=False`).

**P1: baseline.**
1. Từ `PPO/` (chưa sửa gì), đánh giá `../ckpt/PPO/SD1-10x05.pth` (greedy, rồi `aug_sample`) trên tập test SD1 10×5; ghi gap (theo `or_solution`) và thời gian mỗi instance. Lặp lại cho `SD2-10x05.pth`, `SD1-20x10.pth`.
2. Giải quyết ba điểm mở ở P0.
3. Ghi lại kết quả greedy của ReSched trên 20 instance cố định (mốc cho `test_baseline_unchanged`) **trước** khi thêm hook.
**Đạt:** gap greedy của checkpoint SD1 10×5 nằm trong ±1.5 điểm phần trăm so với bài báo (Bảng 10: 11.20% cho PPO ngân sách nhỏ, 11.48% cho ngân sách lớn; chưa rõ checkpoint tương ứng ngân sách nào, hãy ghi lại). Dao động giữa các seed của bài gốc khoảng 1 điểm.

**P2: bộ mô phỏng tham chiếu.** Viết `ref_sim.py` và `test_ref_parity.py`.
**Đạt:** trên 1000 chuỗi hành động ngẫu nhiên (SD1 và SD2, 10×5), makespan của môi trường trùng khớp tuyệt đối với `RefSim`.

**P3: bộ xây dựng quan sát.** Chạy `test_dependency_direction` trước (D2), rồi viết `build_pf_obs` và `ref_obs`.
**Đạt:** `test_dependency_direction.py` qua; `test_obs_builder.py` qua trên 200 trạng thái ngẫu nhiên (SD1, SD2; 10×5 và 20×10) với sai số 1e-5; `xy` luôn trong [0, 1]; chạy được với batch lớn (B × 100) mà không vòng lặp Python theo instance.

**P4: rasterize, readout, U-Net, mô hình.** Viết `pfunet/modules.py` và `pfunet/model.py`.
**Đạt:** `test_rasterize.py`, `test_invariance.py`, `test_policy_interface.py` qua; forward và backward chạy trên RTX 3050 với B = 4, 10×5, `canvas_hw = (64, 64)`, không OOM; đưa trực tiếp vào `PPOTrainer` mà không sửa trainer.

**P5: tích hợp và overfit một instance.** Thêm hook (mục 5.2), chạy `test_baseline_unchanged`, rồi `test_overfit_one.py`.
**Đạt (M3):** `test_baseline_unchanged` qua (ReSched không đổi); sau tối đa 2000 update trên **một** instance 10×5, makespan greedy cách OR-Tools không quá 3% trên chính instance đó.

**P6: smoke train.** Chạy hồ sơ `smoke` 20 đến 50 update trên SD1 10×5, kèm đánh giá greedy.
**Đạt:** không lỗi; loss hữu hạn; điểm validation được ghi; checkpoint lưu và nạp lại được; có số liệu VRAM và thời gian. **Ở mốc này chưa đánh giá chất lượng.**

**P7: pilot trên máy mạnh (cổng quyết định).** Hồ sơ `full`: SD1 và SD2 10×5, ít nhất 4 seed, ngân sách nhỏ rồi lớn. Đánh giá greedy và `aug_sample`.
**Báo cáo:** gap trong phân phối so với ReSched cùng ngân sách, và thời gian mỗi instance.
**Dừng:** nếu gap trong phân phối kém ReSched hơn 3 điểm phần trăm **và** ở 100×10 vẫn không thắng, dừng và báo cáo để chuyển phương án dự phòng.

**P8: quy mô lớn và ablation (máy mạnh).**
- Sinh dữ liệu 100×10, 200×5 (SD1 generator; SD2 cần cấu hình `FlexibleJobShopProblemGenerator`), giải CP-SAT với giới hạn thời gian ghi rõ, ít nhất 20 instance mỗi cỡ.
- Chạy lại ReSched (checkpoint có sẵn và bản tự huấn luyện) trên cùng instance, cùng máy.
- Đường cong gap theo n và **thời gian mỗi bước theo n** cho PF-UNet và ReSched.
- Ablation: bỏ skip, giảm số mức, CNN một mức, readout một mức so với hai mức so với mọi mức, độ phân giải 16/32/64, có và không có kênh chuẩn hóa quy mô `log(J_a)/log(J_max)` (kênh thứ 9, `c_in = 9`), bỏ attention giữa máy, bỏ kênh tải tương lai (kênh 7).
- Ghi nhận trường hợp hết bộ nhớ hoặc quá thời gian như một kết quả.

**P9 (mở rộng): fine-tune tuần tự và baseline kiểu Scale-Net.**
- *Fine-tune tuần tự:* huấn luyện ở 10×5, rồi tinh chỉnh lên 20×10, 40×10, 100×10 với tốc độ học giảm dần mỗi giai đoạn (ví dụ giảm 10 lần); sau mỗi giai đoạn đánh giá lại ở **mọi cỡ nhỏ hơn** để đo hiện tượng quên. So với chế độ zero-shot chỉ huấn luyện ở 10×5.
- *Baseline kiểu Scale-Net (E6b):* cùng MDP và cùng đầu quyết định, nhưng thay phần biểu diễn bằng Graph U-Net trên token operation:

```python
# mã giả baseline: attention trên các operation còn hoạt động, pooling top-k, unpool zero-fill, skip
H0 = embed(ops)                                   # [B, N, d]
for l in range(L):
    Hh[l] = attention(H[l], active[l])            # chỉ trên các nút đang hoạt động
    z = proj(Hh[l]); active[l+1] = topk(z, k[l])  # k[l] = hằng số tuyệt đối (như Scale-Net)
    H[l+1] = Hh[l] gather theo active[l+1]
for l in reversed(range(L)):
    H_up = unpool(H[l+1], active[l], fill=0)      # nút bị bỏ nhận vector 0
    Hf[l] = attention(concat(H_up, Hh[l]))        # skip
ctx = concat(h_cur, Hf[0], summary(Hf[1..L]))     # fusion đa mức vào đầu quyết định
```

  Mục đích: tách "do có U-Net" khỏi "do biểu diễn trường". Dùng một mô hình riêng, không trộn vào PF-UNet.

---

## 8. Quy tắc làm việc

**Phải làm**

- Không sửa logic của ReSched. Chỉ được sửa `PPO/SchedulingRunner.py` và `PPO/configs/fjsp.py` ở mức hook (mục 5.2); không động tới `REINFORCE/`; mọi code mới nằm trong `PPO/pfunet/`. Mặc định (`model_type = 'resched'`) phải cho kết quả y hệt trước khi sửa.
- Mỗi lần chạy ghi: commit, seed (numpy, torch, sinh dữ liệu), cấu hình đầy đủ, phiên bản thư viện, loại GPU, thời gian chạy.
- Chọn siêu tham số và mô hình theo tập validation; **không** dùng tập test để chọn.
- Chạy lại baseline trên cùng máy và cùng instance cho mọi so sánh; không dùng số của bài báo cho cỡ bài chưa báo cáo.
- Báo cáo cả kết quả xấu và lỗi; không đổi tiêu chí để vượt qua.
- Mọi thay đổi so với đề cương (D1 đến D6 và bất kỳ thay đổi nào sau đó) ghi vào `design_changes.md`.

**Phải dừng lại và hỏi**

- Giao diện mô hình–trainer khác với `repo_facts.md`, hoặc `test_dependency_direction` sai cả hai chiều (D2).
- `test_ref_parity` hoặc `test_obs_builder` không qua sau hai lần thử hợp lý.
- Không tái hiện được P1.
- Hàm reward hoặc chuyển trạng thái của môi trường không khớp `RefSim`.
- Một giai đoạn không đạt sau hai lần thử hợp lý.
- Yêu cầu nào làm thay đổi logic môi trường hoặc reward của ReSched.

**Báo cáo cuối mỗi giai đoạn (ngắn):** việc đã làm, kết quả so với tiêu chí đạt (kèm số liệu), tệp đã đổi, điều bất ngờ, câu hỏi còn mở.

---

## 9. Bẫy thường gặp

- **`scatter_` và `scatter_add_`:** khi gán xác suất về `prob [B, N_rem, M]` phải dùng `scatter_add_`. Các slot đệm (không có head) vẫn trỏ tới một hàng operation thật và sẽ ghi 0 đè lên nó nếu dùng `scatter_`.
- **Thứ tự lưới của `grid_sample`:** `grid[..., 0]` là x (chiều rộng), `grid[..., 1]` là y (chiều cao). Đảo thứ tự sẽ làm đọc sai mà không báo lỗi.
- **`align_corners`:** splat và readout phải cùng quy ước (`True`), nếu không có độ lệch nửa pixel.
- **Chỉ số operation đổi sau mỗi bước:** `head_idx` phải tính lại mỗi bước từ `action_mask`, không cache.
- **Dtype và `gather` trên bool:** `rev_pos` là `int32`, không dùng làm chỉ số. Nếu `torch.gather` không hỗ trợ tensor bool ở phiên bản đang dùng, chuyển `dependency_o` sang `uint8` rồi đổi lại bool.
- **Hướng của `dependency_o`:** hàng hay cột của head mới là phần còn lại của job phải được xác nhận bằng `test_dependency_direction`; đừng giả định.
- **`argsort` phải `stable=True`** để các head giữ thứ tự chỉ số operation và kết quả tất định.
- **`.squeeze()` với B = 1:** bản gốc trả về scalar; giữ nguyên hành vi, đừng "sửa".
- **Rollout:** gọi `model_old` trong `torch.no_grad()` và `eval()`; không lưu canvas hay bản đồ U-Net vào bộ đệm.
- **Windows:** đường dẫn tương đối `../data` phụ thuộc thư mục chạy; nếu DataLoader hoặc đa tiến trình gây lỗi, thử `num_workers = 0`.
- **`aug_sample` nhân batch 100 lần:** tách theo `eval_chunk` khi VRAM hạn chế; không giảm mô hình.
- **Seed:** đặt seed riêng cho numpy, torch (CPU và CUDA) và sinh dữ liệu; ghi vào log.
