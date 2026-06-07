"""backward 의 모든 intermediate quantity 의 수치 비교 — 2backward vs (B).

비교 대상:
  - grad_y_main, grad_y_align (= z 의 grad)
  - grad_W_main, grad_W_align (= weight 의 grad, sum 전 별도)
  - adaptive_weight ratio
  - 최종 W.grad, x.grad, bias.grad

설정:
  - autocast(bf16) 안 — 실제 학습 환경
  - same input, same weight init, same align_target
"""
import sys, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix')


# ─── 기존 CausalConv3d (= 2backward 용) ─────────
class CausalConv3d(nn.Conv3d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2*self.padding[0], 0)
        self.padding = (0, 0, 0)
    def forward(self, x, cache_x=None):
        x = F.pad(x, self._padding)
        return super().forward(x)


def main():
    torch.manual_seed(42)
    device = 'cuda:0'
    dtype = torch.float32   # weight init = float32 (= 학습 의 setup)

    # 두 mechanism 의 same weight init 위해
    in_ch, out_ch = 8, 16
    causal_template = CausalConv3d(in_ch, out_ch, 3, padding=1).to(device=device, dtype=dtype)
    init_W = causal_template.weight.detach().clone()
    init_b = causal_template.bias.detach().clone()

    # input + align_target (= same seed)
    x_orig = torch.randn(1, in_ch, 3, 8, 8, device=device, dtype=dtype)
    align_target = torch.randn(1, out_ch, 3, 8, 8, device=device, dtype=dtype)

    alpha = 1.0   # disc_weight (= align_weight)

    # ─── 1. 2backward 의 backward 직접 trace ───────────────
    print("=" * 70)
    print("1. 2backward — autograd.grad × 2 + L_total.backward()")
    print("=" * 70)
    c1 = CausalConv3d(in_ch, out_ch, 3, padding=1).to(device=device, dtype=dtype)
    c1.weight.data.copy_(init_W); c1.bias.data.copy_(init_b)
    x1 = x_orig.detach().clone().requires_grad_(True)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y1 = c1(x1)
        L_rec_1 = (y1 ** 2).mean()
        L_align_1 = (y1 * align_target).mean()

    # autograd.grad — autocast 의 안 또는 밖? autocast 의 결과 cast 위해 안:
    rec_grads_W = torch.autograd.grad(L_rec_1, c1.weight, retain_graph=True)[0]
    align_grads_W = torch.autograd.grad(L_align_1, c1.weight, retain_graph=True)[0]
    print(f"  rec_grads_W.dtype:   {rec_grads_W.dtype},   norm: {rec_grads_W.norm().item():.6e}")
    print(f"  align_grads_W.dtype: {align_grads_W.dtype},   norm: {align_grads_W.norm().item():.6e}")
    ratio_2bwd = (rec_grads_W.norm() / (align_grads_W.norm() + 1e-4)).clamp(0, 1e4)
    c_2bwd = ratio_2bwd.detach() * alpha
    print(f"  adaptive_weight (c): {c_2bwd.item():.6f}")

    L_total_1 = L_rec_1 + c_2bwd * L_align_1
    L_total_1.backward()
    gW_2bwd = c1.weight.grad.clone()
    gx_2bwd = x1.grad.clone()
    gb_2bwd = c1.bias.grad.clone()

    print(f"\n  최종:")
    print(f"    W.grad   norm: {gW_2bwd.norm().item():.6e}, dtype: {gW_2bwd.dtype}")
    print(f"    x.grad   norm: {gx_2bwd.norm().item():.6e}, dtype: {gx_2bwd.dtype}")
    print(f"    b.grad   norm: {gb_2bwd.norm().item():.6e}, dtype: {gb_2bwd.dtype}")

    # ─── 2. (B) 의 backward 직접 trace ─────────────
    print(f"\n" + "=" * 70)
    print("2. (B) — AdaptiveWeightedCausalConv3d 의 single backward (= manual cast 적용)")
    print("=" * 70)
    from adaptive_weighted_causal_conv_3d import AdaptiveWeightedCausalConv3d, _AdaptiveWeightedConv3dFn

    c2 = AdaptiveWeightedCausalConv3d(in_ch, out_ch, 3, padding=1,
                                       disc_weight=alpha, adaptive_weight_eps=1e-4,
                                       adaptive_weight_max=1e4).to(device=device, dtype=dtype)
    c2.weight.data.copy_(init_W); c2.bias.data.copy_(init_b)
    x2 = x_orig.detach().clone().requires_grad_(True)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y_main, y_adv = c2(x2)
        L_rec_2 = (y_main ** 2).mean()
        L_align_2 = (y_adv * align_target).mean()
        L_total_2 = L_rec_2 + L_align_2

    L_total_2.backward()
    gW_b = c2.weight.grad.clone()
    gx_b = x2.grad.clone()
    gb_b = c2.bias.grad.clone()
    c_b = _AdaptiveWeightedConv3dFn._last_c

    print(f"  adaptive_weight (c): {c_b.item():.6f}")
    print(f"\n  최종:")
    print(f"    W.grad   norm: {gW_b.norm().item():.6e}, dtype: {gW_b.dtype}")
    print(f"    x.grad   norm: {gx_b.norm().item():.6e}, dtype: {gx_b.dtype}")
    print(f"    b.grad   norm: {gb_b.norm().item():.6e}, dtype: {gb_b.dtype}")

    # ─── 3. 비교 ─────────────────────────────────
    print(f"\n" + "=" * 70)
    print("3. 직접 수치 비교")
    print("=" * 70)
    print(f"  adaptive_weight diff: {abs(c_2bwd.item() - c_b.item()):.4e}")
    print(f"  W.grad   max diff: {(gW_2bwd - gW_b).abs().max().item():.4e},  rel: {(gW_2bwd - gW_b).abs().max() / gW_2bwd.abs().max():.4e}")
    print(f"  x.grad   max diff: {(gx_2bwd - gx_b).abs().max().item():.4e},  rel: {(gx_2bwd - gx_b).abs().max() / gx_2bwd.abs().max():.4e}")
    print(f"  b.grad   max diff: {(gb_2bwd - gb_b).abs().max().item():.4e},  rel: {(gb_2bwd - gb_b).abs().max() / gb_2bwd.abs().max():.4e}")

    # 동등 성 판정
    def ok(d, r, tol_abs=1e-3, tol_rel=1e-2):
        return d < tol_abs or r < tol_rel
    print(f"\n  ✅ 동등 (rel < 1%):" if ok((gW_2bwd - gW_b).abs().max().item(), ((gW_2bwd - gW_b).abs().max() / gW_2bwd.abs().max()).item()) else f"  ❌ 다름")


if __name__ == "__main__":
    main()
