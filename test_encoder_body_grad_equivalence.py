"""Encoder body grad equivalence test — (B) vs 2backward.

목적:
  - 새 (B) 식 (= AdaptiveWeightedCausalConv3d) 의 backward 시 encoder body 의 grad 가
    2backward 의 grad 와 동등 한지 검증.
  - 단 single 1-step test 만 — 학습 의 N step 의 trend 의 검증 X.

기준:
  - same input + same weight init
  - L_rec = (y_main ** 2).mean()
  - L_align = (y_adv * align_target).mean()
  - disc_weight (= align_weight) = 1.0

비교:
  - encoder body (= self.head[-1] 외 의 다른 layer) 의 W.grad
  - 두 mechanism 의 동등 성 (= allclose) 확인
"""
import sys, torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

# kinemadae_geoprior 의 Encoder3d 만
spec = importlib.util.spec_from_file_location(
    "kinemadae_geoprior",
    "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/kinemadae_geoprior.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules['kinemadae_geoprior'] = mod
spec.loader.exec_module(mod)

Encoder3d = mod.Encoder3d


def run_encoder_b_adaptive(x, align_target, init_state, alpha=1.0):
    """(B) 식 — single backward 의 AdaptiveWeightedCausalConv3d."""
    enc = Encoder3d(dim=32, z_dim=32, dim_mult=[1, 2], num_res_blocks=1,
                    temperal_downsample=[True],
                    use_b_adaptive=True,
                    b_adaptive_disc_weight=alpha,
                    b_adaptive_eps=1e-4,
                    b_adaptive_max=1e4).float().train()
    enc.load_state_dict(init_state, strict=True)

    x_in = x.detach().clone().requires_grad_(True)
    y_main, y_adv = enc(x_in)
    L_rec = (y_main ** 2).mean()
    L_align = (y_adv * align_target).mean()
    L_total = L_rec + L_align    # alpha 가 backward 안 에서 적용
    L_total.backward()

    # 모든 layer 의 grad 수집
    grads = {name: p.grad.detach().clone() for name, p in enc.named_parameters() if p.grad is not None}
    return grads, x_in.grad.detach().clone()


def run_encoder_2backward(x, align_target, init_state, alpha=1.0):
    """2backward 식 — 기존 CausalConv3d + autograd.grad × 2."""
    enc = Encoder3d(dim=32, z_dim=32, dim_mult=[1, 2], num_res_blocks=1,
                    temperal_downsample=[True],
                    use_b_adaptive=False).float().train()
    # init_state 의 head.2 key 는 새 class 의 weight, bias 의 key 와 동일 — 그대로 load
    enc.load_state_dict(init_state, strict=True)

    x_in = x.detach().clone().requires_grad_(True)
    y = enc(x_in)
    L_rec   = (y ** 2).mean()
    L_align = (y * align_target).mean()

    # ratio 측정 — encoder.head[-1].weight 의 grad
    W = enc.head[-1].weight
    rec_grads   = torch.autograd.grad(L_rec,   W, retain_graph=True)[0]
    align_grads = torch.autograd.grad(L_align, W, retain_graph=True)[0]
    w = torch.norm(rec_grads) / (torch.norm(align_grads) + 1e-4)
    w = w.clamp(0.0, 1e4)
    c = w.detach() * alpha

    L_total = L_rec + c * L_align
    L_total.backward()

    grads = {name: p.grad.detach().clone() for name, p in enc.named_parameters() if p.grad is not None}
    return grads, x_in.grad.detach().clone(), c


def test_encoder_body_grad_equivalence():
    """encoder body 의 모든 layer 의 grad 가 (B) vs 2backward 의 동등 한지."""
    print("\n[Test] encoder body grad equivalence — (B) vs 2backward")
    torch.manual_seed(42)

    # 두 mechanism 의 same init 위해 — 한 번 init 후 share
    enc_template = Encoder3d(dim=32, z_dim=32, dim_mult=[1, 2], num_res_blocks=1,
                              temperal_downsample=[True],
                              use_b_adaptive=False).float()
    init_state = {k: v.clone() for k, v in enc_template.state_dict().items()}

    x = torch.randn(1, 3, 5, 8, 8)
    # output shape 측정 위해 dummy forward
    with torch.no_grad():
        y_shape = enc_template(x).shape
    align_target = torch.randn(*y_shape)

    # (B) 식
    grads_b, gx_b = run_encoder_b_adaptive(x, align_target, init_state, alpha=1.0)
    # 2backward 식
    grads_2bwd, gx_2bwd, c_2bwd = run_encoder_2backward(x, align_target, init_state, alpha=1.0)

    print(f"  2backward 의 adaptive_weight (c): {c_2bwd.item():.6f}")

    # 비교
    print(f"\n  === parameter 별 grad diff ===")
    print(f"  {'name':<45} {'(B) ‖grad‖':>14} {'2bwd ‖grad‖':>14} {'max diff':>12} {'rel':>10}")
    print(f"  {'-'*45} {'-'*14} {'-'*14} {'-'*12} {'-'*10}")
    body_ok = True
    for name in sorted(grads_b.keys()):
        g_b = grads_b[name]
        g_2 = grads_2bwd[name]
        max_d = (g_b - g_2).abs().max().item()
        n_b = g_b.norm().item()
        n_2 = g_2.norm().item()
        rel = max_d / max(n_2, 1e-10)
        status = '✅' if rel < 1e-3 else '❌'
        print(f"  {name:<45} {n_b:14.4f} {n_2:14.4f} {max_d:12.2e} {rel:10.2e} {status}")
        if rel > 1e-3:
            body_ok = False

    print(f"\n  x.grad max diff: {(gx_b - gx_2bwd).abs().max().item():.2e}")
    print(f"\n  {'✅ encoder body grad 모두 동등' if body_ok else '❌ 일부 layer 의 grad 다름'}")


if __name__ == "__main__":
    print("=" * 70)
    print("Encoder body grad equivalence test — (B) vs 2backward")
    print("=" * 70)
    test_encoder_body_grad_equivalence()
    print("=" * 70)
