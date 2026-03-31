#!/usr/bin/env python3
"""
快速验证 Sparse24PenaltyFunction：前向恒等、反向与显式 penalty 一致、可优化器更新。
无需跑完整 LLM 训练或等到 alpha/sparse_reg 触发。

优先从 change_to_relu_ddp2 导入（与线上一致）；若因缺少 datasets/accelerate 等失败，
则使用下方内联实现 —— 请与 change_to_relu_ddp2 中对应函数保持同步。

用法（在仓库根目录）:
  python verify_sparse24_penalty.py
  CUDA_VISIBLE_DEVICES=0 python verify_sparse24_penalty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

_SOURCE = "change_to_relu_ddp2"

try:
    from change_to_relu_ddp2 import Sparse24PenaltyFunction, sparse24_activation_penalty
except (ImportError, ModuleNotFoundError) as e:
    _SOURCE = f"inlined (import failed: {e})"

    def sparse24_activation_penalty(h: torch.Tensor) -> torch.Tensor:
        x = h.reshape(-1, h.shape[-1])
        K = x.shape[1]
        pad = (4 - K % 4) % 4
        if pad:
            x = F.pad(x, (0, pad))
        groups = x.reshape(-1, 4)
        abs_g = groups.abs()
        _, bot2_idx = abs_g.topk(2, dim=1, largest=False, sorted=False)
        penalty = groups.gather(1, bot2_idx).abs().mean()
        return penalty

    class Sparse24PenaltyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h: torch.Tensor, coeff: float) -> torch.Tensor:
            ctx.coeff = float(coeff)
            ctx.save_for_backward(h)
            return h

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            h, = ctx.saved_tensors
            coeff = ctx.coeff
            with torch.enable_grad():
                h_det = h.detach().requires_grad_(True)
                pen = sparse24_activation_penalty(h_det)
                grad_pen, = torch.autograd.grad(
                    pen,
                    h_det,
                    retain_graph=False,
                    create_graph=False,
                )
            return grad_output + coeff * grad_pen, None


def _dtype_for_device(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def test_forward_identity(device: torch.device) -> None:
    dt = _dtype_for_device(device)
    h = torch.randn(2, 8, 64, device=device, dtype=dt, requires_grad=True)
    coeff = 0.001
    out = Sparse24PenaltyFunction.apply(h, coeff)
    assert out.shape == h.shape
    assert torch.equal(out, h), "forward 应为恒等映射"


def test_backward_matches_explicit_penalty(device: torch.device) -> None:
    dt = _dtype_for_device(device)
    coeff = 0.01
    torch.manual_seed(0)
    h0 = torch.randn(3, 16, 128, device=device, dtype=dt)

    h_a = h0.clone().requires_grad_(True)
    out_a = Sparse24PenaltyFunction.apply(h_a, coeff)
    loss_a = out_a.float().sum()
    loss_a.backward()
    grad_a = h_a.grad.float()

    h_b = h0.clone().requires_grad_(True)
    loss_b = h_b.float().sum() + coeff * sparse24_activation_penalty(h_b)
    loss_b.backward()
    grad_b = h_b.grad.float()

    if not torch.allclose(grad_a, grad_b, rtol=1e-2, atol=1e-2):
        diff = (grad_a - grad_b).abs().max().item()
        raise AssertionError(f"梯度与显式 penalty 不一致, max|diff|={diff}")
    assert torch.isfinite(grad_a).all(), "梯度含 NaN/Inf"


def test_optimizer_step(device: torch.device) -> None:
    dt = _dtype_for_device(device)
    lin = nn.Linear(32, 64, device=device, dtype=dt)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    x = torch.randn(4, 32, device=device, dtype=dt)
    coeff = 0.001

    opt.zero_grad()
    h = lin(x)
    h = Sparse24PenaltyFunction.apply(h, coeff)
    loss = (h.float() ** 2).mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(next(lin.parameters())).all()


def test_coeff_zero_skipped_in_mlp_style(device: torch.device) -> None:
    dt = _dtype_for_device(device)
    h = torch.randn(2, 8, 32, device=device, dtype=dt, requires_grad=True)
    coeff = 0.0
    out = h if coeff <= 0.0 else Sparse24PenaltyFunction.apply(h, coeff)
    out.float().sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"实现来源: {_SOURCE}")
    print(f"设备: {device}")

    test_forward_identity(device)
    print("  [ok] forward 恒等")

    test_backward_matches_explicit_penalty(device)
    print("  [ok] backward 与显式 penalty 一致")

    test_optimizer_step(device)
    print("  [ok] AdamW 一步更新")

    test_coeff_zero_skipped_in_mlp_style(device)
    print("  [ok] coeff=0 路径（不调用 apply）")

    print("\nSparse24PenaltyFunction 验证全部通过。")


if __name__ == "__main__":
    main()
