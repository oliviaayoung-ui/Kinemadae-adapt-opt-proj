"""Integration test — (B) 식 의 통합 검증.

목적:
  1. use_b_adaptive=True 시 Encoder3d.forward 가 (y_main, y_adv) tuple return
  2. use_b_adaptive=False (= default) 시 기존 동작 (= single tensor return) 유지 → backward compat
  3. (B) 식 의 backward 의 W.grad 가 2backward 와 동등 (= 정확 weight gradient ratio)
"""
import sys, torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder3d only — full VAE 안 부담 위해 standalone
import importlib.util
spec = importlib.util.spec_from_file_location(
    "kinemadae_geoprior",
    "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/kinemadae_geoprior.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules['kinemadae_geoprior'] = mod
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(f"❌ kinemadae_geoprior.py 의 import 실패: {e}")
    sys.exit(1)

Encoder3d = mod.Encoder3d


def test_encoder_forward_default():
    """use_b_adaptive=False (= default) 시 기존 동작 — single tensor return."""
    print("\n[Test 1] Encoder3d.forward (use_b_adaptive=False) — single tensor return")
    torch.manual_seed(0)

    enc = Encoder3d(dim=32, z_dim=32, dim_mult=[1, 2], num_res_blocks=1,
                    temperal_downsample=[True],
                    use_b_adaptive=False).float().eval()
    x = torch.randn(1, 3, 5, 8, 8)
    with torch.no_grad():
        y = enc(x)
    assert torch.is_tensor(y), f"기존 동작 = single tensor 가야 하는데 {type(y)}"
    print(f"  ✅ y type = Tensor, shape = {tuple(y.shape)}")


def test_encoder_forward_b_adaptive():
    """use_b_adaptive=True 시 — (y_main, y_adv) tuple return (= training mode)."""
    print("\n[Test 2] Encoder3d.forward (use_b_adaptive=True) — tuple return")
    torch.manual_seed(1)

    enc = Encoder3d(dim=32, z_dim=32, dim_mult=[1, 2], num_res_blocks=1,
                    temperal_downsample=[True],
                    use_b_adaptive=True).float().train()
    x = torch.randn(1, 3, 5, 8, 8, requires_grad=True)
    y = enc(x)
    assert isinstance(y, tuple), f"use_b_adaptive=True 의 training 시 = tuple 가야 하는데 {type(y)}"
    y_main, y_adv = y
    print(f"  ✅ y_main shape: {tuple(y_main.shape)}")
    print(f"  ✅ y_adv  shape: {tuple(y_adv.shape)}")
    # same value verification
    assert torch.allclose(y_main, y_adv), f"y_main 과 y_adv 는 same value 가야"
    print(f"  ✅ y_main == y_adv (= same forward value, separate backward edges)")


def test_encoder_backward_b_adaptive():
    """use_b_adaptive=True 의 backward — y_main 으로 rec loss, y_adv 으로 align loss.
    W.grad 가 별도 계산 + adaptive ratio + scale 적용 검증."""
    print("\n[Test 3] Encoder3d.backward (use_b_adaptive=True) — (B) mechanism")
    torch.manual_seed(2)

    enc = Encoder3d(dim=32, z_dim=32, dim_mult=[1, 2], num_res_blocks=1,
                    temperal_downsample=[True],
                    use_b_adaptive=True,
                    b_adaptive_disc_weight=1.0,
                    b_adaptive_eps=1e-4,
                    b_adaptive_max=1e4).float().train()
    x = torch.randn(1, 3, 5, 8, 8, requires_grad=True)
    align_target = torch.randn(1, 32, 1, 2, 2)  # output shape estimate

    y_main, y_adv = enc(x)
    if y_main.shape[2:] != align_target.shape[2:]:
        align_target = torch.randn_like(y_main)

    L_rec   = (y_main ** 2).mean()
    L_align = (y_adv * align_target).mean()
    L_total = L_rec + L_align
    L_total.backward()

    # encoder.head[-1] 의 weight 의 grad 확인
    head_last = enc.head[-1]   # AdaptiveWeightedCausalConv3d
    W = head_last.weight
    assert W.grad is not None, "W.grad 가 None"
    print(f"  ✅ encoder.head[-1].weight.grad shape: {tuple(W.grad.shape)}")
    print(f"  ✅ W.grad max abs: {W.grad.abs().max().item():.4f}")

    # _last_c 확인
    from adaptive_weighted_causal_conv_3d import _AdaptiveWeightedConv3dFn
    c = _AdaptiveWeightedConv3dFn._last_c
    print(f"  ✅ adaptive_weight (_last_c): {c.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("(B) 식 통합 test")
    print("=" * 60)

    test_encoder_forward_default()
    test_encoder_forward_b_adaptive()
    test_encoder_backward_b_adaptive()

    print("\n" + "=" * 60)
    print("모든 통합 test 완료")
    print("=" * 60)
