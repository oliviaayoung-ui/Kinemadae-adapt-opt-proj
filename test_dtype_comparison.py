"""autocast(bf16) 안 의 2backward vs (B) 의 dtype trace 비교.

목적:
  - 2backward 의 backward 시 의 saved tensor, grad_output 의 dtype
  - (B) 의 backward 시 의 saved tensor, grad_output 의 dtype
  - 두 가지 의 일치 / mismatch 확인
"""
import sys, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix')
from adaptive_weighted_causal_conv_3d import AdaptiveWeightedCausalConv3d, _AdaptiveWeightedConv3dFn


class CausalConv3d(nn.Conv3d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2*self.padding[0], 0)
        self.padding = (0, 0, 0)
    def forward(self, x, cache_x=None):
        x = F.pad(x, self._padding)
        return super().forward(x)


# ─── Test 1: 2backward 의 dtype trace ─────────────
print("=" * 60)
print("Test 1: 2backward (= nn.Conv3d + autograd.grad) 의 dtype")
print("=" * 60)

torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
causal = CausalConv3d(8, 16, 3, padding=1).to(device)
x = torch.randn(1, 8, 3, 8, 8, device=device, requires_grad=True)

print(f"\nweight dtype: {causal.weight.dtype}")
print(f"input  dtype: {x.dtype}")
print(f"\n>>> autocast(bf16) context 안 의 forward")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    y = causal(x)
    print(f"  y (= forward output) dtype: {y.dtype}")
    L_rec = (y ** 2).mean()
    L_align = (y * torch.randn_like(y)).mean()
    print(f"  L_rec   dtype: {L_rec.dtype}")
    print(f"  L_align dtype: {L_align.dtype}")

    # autograd.grad 의 backward
    rec_grads = torch.autograd.grad(L_rec, causal.weight, retain_graph=True)[0]
    align_grads = torch.autograd.grad(L_align, causal.weight, retain_graph=True)[0]
    print(f"\n  autograd.grad(L_rec, weight) dtype: {rec_grads.dtype}")
    print(f"  autograd.grad(L_align, weight) dtype: {align_grads.dtype}")
    print(f"  weight dtype (= 학습 의 update 의 의도): {causal.weight.dtype}")


# ─── Test 2: (B) 의 dtype trace (= manual print) ─────────────
print("\n" + "=" * 60)
print("Test 2: (B) 의 dtype trace (= custom autograd.Function 안)")
print("=" * 60)

# (B) 의 backward 안 의 print 추가 위해 임시 wrapper class
class _PrintFn(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        # [FIX] manual cast — autocast 의 dtype 으로
        if torch.is_autocast_enabled():
            _ac_dtype = torch.get_autocast_gpu_dtype()
            if x.dtype != _ac_dtype:
                x = x.to(_ac_dtype)
            if weight.dtype != _ac_dtype:
                weight = weight.to(_ac_dtype)
            if bias is not None and bias.dtype != _ac_dtype:
                bias = bias.to(_ac_dtype)
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.zeros(1))
        ctx.has_bias = bias is not None
        ctx.meta = (stride, padding, dilation, groups)
        y = F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        print(f"  [forward] x.dtype: {x.dtype}, weight.dtype: {weight.dtype}, y.dtype: {y.dtype}")
        return y, y.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_y_main, grad_y_adv):
        x, weight, bias = ctx.saved_tensors
        print(f"  [backward]")
        print(f"    ctx.saved x.dtype:      {x.dtype}     (= ctx.save 시 저장 된 dtype)")
        print(f"    ctx.saved weight.dtype: {weight.dtype}")
        print(f"    grad_y_main.dtype:      {grad_y_main.dtype}    (= backward 의 incoming grad)")
        print(f"    grad_y_adv.dtype:       {grad_y_adv.dtype if grad_y_adv is not None else 'None'}")
        # autograd.Function 의 return = 4 None (= input 의 grad)
        return torch.zeros_like(x), torch.zeros_like(weight), None, None, None, None, None

torch.manual_seed(0)
causal_b = CausalConv3d(8, 16, 3, padding=1).to(device)
x_b = torch.randn(1, 8, 3, 8, 8, device=device, requires_grad=True)

print(f"\nweight dtype: {causal_b.weight.dtype}")
print(f"input  dtype: {x_b.dtype}")
print(f"\n>>> autocast(bf16) context 안 의 forward + backward")
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # _PrintFn 직접 호출 (= AdaptiveWeightedCausalConv3d 의 inside mechanism 의 simulation)
    x_padded = F.pad(x_b, causal_b._padding)
    y_main, y_adv = _PrintFn.apply(x_padded, causal_b.weight, causal_b.bias,
                                    causal_b.stride, (0,0,0), causal_b.dilation, causal_b.groups)
    L_rec = (y_main ** 2).mean()
    L_align = (y_adv * torch.randn_like(y_adv)).mean()
    L_total = L_rec + L_align
    L_total.backward()
