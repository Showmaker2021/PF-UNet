import sys
import os
import torch
import numpy as np

ppo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ppo_dir not in sys.path:
    sys.path.insert(0, ppo_dir)

from pfunet.modules import rasterize, readout

def test_rasterize_and_readout():
    print("Testing rasterize and readout unit tests...")
    H, W = 64, 64
    B, M, J, C = 1, 1, 1, 1
    
    # 1. One point exactly at pixel center (e.g. pixel (10, 20))
    # gx = x * (W - 1) -> x = 10 / (W - 1)
    # gy = y * (H - 1) -> y = 20 / (H - 1)
    px, py = 10, 20
    xy_center = torch.tensor([[[[px / (W - 1), py / (H - 1)]]]], dtype=torch.float32)
    vals = torch.ones((B, M, J, C), dtype=torch.float32)
    mask = torch.ones((B, M, J), dtype=torch.bool)
    
    canvas = rasterize(xy_center, vals, mask, H, W)  # [1, 1, H, W]
    assert canvas[0, 0, py, px].item() == 1.0, f"Expected 1.0 at ({py}, {px}), got {canvas[0, 0, py, px].item()}"
    assert canvas.sum().item() == 1.0, f"Expected total sum 1.0 (one-hot), got {canvas.sum().item()}"
    print("[PASS] One-point at pixel center is exactly one-hot.")
    
    # 2. One point exactly at midpoint between (10, 20) and (11, 21)
    xy_mid = torch.tensor([[[[(px + 0.5) / (W - 1), (py + 0.5) / (H - 1)]]]], dtype=torch.float32)
    canvas_mid = rasterize(xy_mid, vals, mask, H, W)
    for y_i, x_i in ((py, px), (py, px + 1), (py + 1, px), (py + 1, px + 1)):
        val = canvas_mid[0, 0, y_i, x_i].item()
        assert abs(val - 0.25) < 1e-6, f"Expected 0.25 at ({y_i}, {x_i}), got {val}"
    assert abs(canvas_mid.sum().item() - 1.0) < 1e-6, f"Total mass not conserved: {canvas_mid.sum().item()}"
    print("[PASS] One-point at midpoint distributes 0.25 to 4 neighboring pixels.")
    
    # 3. Mass conservation across random multi-point configuration
    B_rnd, M_rnd, J_rnd = 4, 5, 10
    xy_rnd = torch.rand((B_rnd, M_rnd, J_rnd, 2), dtype=torch.float32)
    vals_rnd = torch.ones((B_rnd, M_rnd, J_rnd, 1), dtype=torch.float32)
    mask_rnd = torch.rand((B_rnd, M_rnd, J_rnd)) > 0.3
    
    canvas_rnd = rasterize(xy_rnd, vals_rnd, mask_rnd, H, W)
    # Sum over canvas spatial dimensions H, W
    splat_sums = canvas_rnd.view(B_rnd, M_rnd, 1, H * W).sum(-1).squeeze(-1)  # [B, M]
    expected_sums = mask_rnd.float().sum(-1)  # [B, M]
    assert torch.allclose(splat_sums, expected_sums, atol=1e-5), \
        f"Mass conservation failed: max diff = {(splat_sums - expected_sums).abs().max()}"
    print("[PASS] Mass conservation holds across arbitrary multi-point batches.")
    
    # 4. Readout of constant feature map
    const_val = 3.14159
    const_map = torch.full((B_rnd * M_rnd, 16, H, W), const_val, dtype=torch.float32)
    readout_vals = readout(const_map, xy_rnd, B_rnd, M_rnd)  # [B, M, J, 16]
    assert torch.allclose(readout_vals, torch.tensor(const_val), atol=1e-5), \
        f"Readout of constant map failed: max diff = {(readout_vals - const_val).abs().max()}"
    print("[PASS] Readout of constant map returns constant everywhere.")
    
    print("\n[SUCCESS] All rasterize and readout unit tests passed!")

if __name__ == '__main__':
    test_rasterize_and_readout()
