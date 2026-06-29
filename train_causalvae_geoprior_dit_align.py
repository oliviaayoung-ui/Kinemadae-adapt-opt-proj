# [NEW - oliviaa/dit_align] Geoprior VAE + DiT intermediate feature alignment training.
#
# 기존 geoprior VAE 학습에 DiT feature alignment loss 추가.
# Teacher: pretrained Wan VAE → pretrained DiT (all frozen)
# Student: geoprior z_cat (48ch) → expanded patchify → same DiT blocks (grad checkpoint)
# Alignment: per-layer feature MSE/cosine after parameter-free upsample.
#
# 가져온 것:
#   [COPIED] train_causalvae.py — 전체 training infrastructure
#   [COPIED] train_causalvae_geoprior.py — geoprior args pre-parse + _video_vae patch
#   [COPIED] dit_feature_extractor.py — load_pipeline, prepare_y, prepare_clip_feature, prepare_null_context
#   [COPIED] WanModel.forward() 구조 — blocks 순회 + gradient_checkpoint_forward
#
# 새로 작성한 것:
#   [NEW] GeopriorDiTAlignModel — forward에서 z_cat 노출하는 wrapper
#   [NEW] compute_alignment_loss — per-layer feature alignment (MSE/cosine)
#   [NEW] create_student_patchify — DiT patchify 확장 (36ch→104ch)
#   [NEW] dit_forward_with_features — blocks 순회하면서 features 수집

# ─── Geoprior pre-parse ──────────────────────────────────────
# [COPIED from train_causalvae_geoprior.py:1-55]
# geoprior 전용 args를 먼저 파싱 후 sys.argv에서 제거.
# _video_vae → _video_vae_geoprior로 패치.
import argparse
import sys
from functools import partial
import json as _json

import kinemadae_geoprior as kinemadae

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--subsample_mode', default='avg_pool',
                     choices=['avg_pool', 'stride', 'bilinear'])
_parser.add_argument('--no_dual_branch', action='store_true')
_parser.add_argument('--prior_z_dim', type=int, default=16)
_parser.add_argument('--add_decoder_tail_stages', type=str, default=None)
_parser.add_argument('--add_decoder_before_head_stages', type=str, default=None)
_parser.add_argument('--decoder_conv1_zmain_init', type=str, default='zero',
                     choices=['zero', 'pretrained'])
_parser.add_argument('--no_expand_conv2', action='store_true')
_parser.add_argument('--expand_encoder_head', action='store_true')
_parser.add_argument('--use_b_adaptive', action='store_true', default=False,
                     help="[NEW - oliviaa/B-fix] encoder.head[-1] 의 (B) 식 mechanism")
# [RESTORE] --adaptive_max_weight 를 b_adaptive clamp(b_adaptive_max)로 전달.
#   이전엔 b_adaptive 경로가 하드코딩 1e7 만 썼음(=clamp1e5fix 유실). 이걸 partial 로 넘겨 conv clamp 에 적용.
_parser.add_argument('--adaptive_max_weight', type=float, default=1e4,
                     help="[RESTORE] b_adaptive clamp max (geoprior AdaptiveWeightedCausalConv3d). 1x52jvfx=1e5(100000)")
_known, _remaining = _parser.parse_known_args()
# [FIX - oliviaa/B-fix] _parser 가 --use_b_adaptive 의 추출 → sys.argv 에서 제거 →
# main parser 가 받지 못함 → args.use_b_adaptive = False (= 학습 의 다른 부분 의 verify log fail)
# 해결: _remaining 에 --use_b_adaptive 다시 inject — 두 parser 모두 처리
sys.argv = [sys.argv[0]] + _remaining
# main parser 도 --use_b_adaptive 의 받기 위해 re-inject
if _known.use_b_adaptive:
    sys.argv.append('--use_b_adaptive')
# [RESTORE] adaptive_max_weight 도 main parser 가 받게 re-inject (pre-parser 가 consume 했으므로)
sys.argv += ['--adaptive_max_weight', str(_known.adaptive_max_weight)]

dual_branch = not _known.no_dual_branch
_tail_stages = _json.loads(_known.add_decoder_tail_stages) if _known.add_decoder_tail_stages else None
_before_head_stages = _json.loads(_known.add_decoder_before_head_stages) if _known.add_decoder_before_head_stages else None
kinemadae._video_vae = partial(
    kinemadae._video_vae_geoprior,
    dual_branch=dual_branch,
    subsample_mode=_known.subsample_mode,
    prior_z_dim=_known.prior_z_dim,
    add_decoder_tail_stages=_tail_stages,
    add_decoder_before_head_stages=_before_head_stages,
    decoder_conv1_zmain_init=_known.decoder_conv1_zmain_init,
    expand_conv2=not _known.no_expand_conv2,
    expand_encoder_head=_known.expand_encoder_head,
    use_b_adaptive=_known.use_b_adaptive,  # [NEW - oliviaa/B-fix]
    b_adaptive_max=_known.adaptive_max_weight,  # [RESTORE] clamp 값 전달 (1x52jvfx=1e5, 이전 하드코딩 1e7 제거)
)
sys.modules['kinemadae'] = kinemadae

# ─── Imports ─────────────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False


from dit_align import (
    compute_alignment_loss, create_student_patchify,
    load_pipeline, prepare_null_context,
    FlowMatchScheduler,
    run_teacher_forward, run_student_forward, setup_dit_memory,
    setup_text_encoder_memory,
    fused_dit_align_forward,
    run_student_diffusion_forward,
)

from torch.utils.data import DataLoader, DistributedSampler, Subset
from PIL import Image
import logging
from colorlog import ColoredFormatter
import tqdm
from itertools import chain
import wandb
from typing import Union, Tuple
import random
import numpy as np
from pathlib import Path
from einops import rearrange
import time

try:
    import lpips
except:
    raise Exception("Need lpips to valid.")

# [COPIED from train_causalvae.py] Local imports
from kinemadae import WanVAE_, _video_vae
from perceptual_loss import LPIPSWithDiscriminator3D
from ema_model import EMA
from ddp_sampler import CustomDistributedSampler, MixedLengthBatchSampler
from video_dataset import TrainVideoDataset, ValidVideoDataset
from video_utils import tensor_to_video
from distrib_utils import DiagonalGaussianDistribution

# [Modified - oliviaa] External library paths
#   Default: <repo_root>/external/DiffSynth-Studio  and  <repo_root>/external/dit_probing
#   Override via env var: KINEMADAE_DIFFSYNTH_PATH, KINEMADAE_PROBING_PATH
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIFFSYNTH = os.environ.get(
    "KINEMADAE_DIFFSYNTH_PATH",
    os.path.join(_REPO_ROOT, "external", "DiffSynth-Studio"),
)
if _DIFFSYNTH not in sys.path:
    sys.path.insert(0, _DIFFSYNTH)
_PROBING = os.environ.get(
    "KINEMADAE_PROBING_PATH",
    os.path.join(_REPO_ROOT, "external", "dit_probing"),
)
if _PROBING not in sys.path:
    sys.path.insert(0, _PROBING)


# ─── NEW: DiT alignment 전용 클래스/함수 ─────────────────────

class GeopriorDiTAlignModel(nn.Module):
    """[NEW - oliviaa/dit_align] Geoprior VAE wrapper.
    kinemadae_geoprior.py forward() 가 z_cat 을 반환하지 않아서 wrapper 필요.
    encode → reparameterize → _encode_prior → cat → decode 를 직접 호출하여 z_cat 노출.
    student_patchify 는 DDP 밖에서 별도 관리 (forward 밖에서 사용되므로 DDP 충돌 방지).
    """
    # pretrained Wan2.1 VAE z_prior 통계 (frozen prior encoder 출력용)
    _prior_mean = torch.tensor([-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
                                 0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921])
    _prior_inv_std = torch.tensor([1.0/2.8184, 1.0/1.4541, 1.0/2.3275, 1.0/2.6558, 1.0/1.2196, 1.0/1.7708,
                                    1.0/2.6052, 1.0/2.0743, 1.0/3.2687, 1.0/2.1526, 1.0/2.8652, 1.0/1.5579,
                                    1.0/1.6382, 1.0/1.1253, 1.0/2.8251, 1.0/1.9160])

    def __init__(self, vae, normalize_zprior=False, zmain_stats=None,
                 decoder_noise_tau_main=0.0, decoder_noise_tau_prior=0.0,
                 decoder_noise_random_mode=True,
                 decoder_noise_warmup_steps=0,
                 decoder_noise_warmup_power=1.0,
                 align_weight=0.0, align_adaptive_weight=False,
                 use_2backward_adaptive=False, adaptive_max_weight=1e4,
                 use_b_adaptive=False,  # [NEW - oliviaa/B-fix] single backward path 의 weight gradient ratio mechanism
                 normalize_zmain_bn=False, bn_momentum=0.1, zmain_bn_init='zprior',
                 z_dim=16,
                 use_align_projection=False, align_proj_dim=5120, align_proj_num_blocks=40,
                 align_projection_init='zero', align_proj_bottleneck_dim=64):
        super().__init__()
        self.vae = vae
        # [NEW - oliviaa] block 별 residual projection (1x1 bottleneck Conv3d = MLP 와 동일, zero init).
        # 구조: Conv3d(D, mid, 1) → Conv3d(mid, D, 1). mid=16 → params ~164K per block (40 block → 6.6M total).
        # REPA/iREPA 등 distillation work 의 표준 patterns (= 1x1 channel projection, spatial context X).
        # spatial context 는 LoRA + patchify 가 main 학습 담당. projection 은 channel re-projection 만.
        # zero init: up conv (= 두 번째) 만 zero → residual 시작 시 trilinear 결과 그대로.
        if use_align_projection:
            mid = align_proj_bottleneck_dim
            self.align_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(align_proj_dim, mid, kernel_size=1),
                    nn.Conv3d(mid, align_proj_dim, kernel_size=1),
                )
                for _ in range(align_proj_num_blocks)
            ])
            if align_projection_init == 'zero':
                # up conv 만 zero (down conv 는 default Kaiming-like init 유지)
                for seq in self.align_projections:
                    nn.init.zeros_(seq[1].weight)
                    nn.init.zeros_(seq[1].bias)
        else:
            self.align_projections = None
        self.normalize_zprior = normalize_zprior
        if normalize_zprior:
            self.register_buffer('prior_mean', self._prior_mean.clone())
            self.register_buffer('prior_inv_std', self._prior_inv_std.clone())
        # [NEW - oliviaa] 변종 B: z_main도 precomputed stats로 정규화 후 alignment에 흘림.
        # 정규화된 z_main을 student_patchify에 입력 → patchify가 normed 분포에 calibrate
        # → Stage 2에서 같은 정규화 적용 시 patchify 직접 reuse 가능 (수학 변환 불필요).
        self.normalize_zmain = zmain_stats is not None
        if self.normalize_zmain:
            zm_mean = torch.tensor(zmain_stats['mean'], dtype=torch.float32)
            zm_std  = torch.tensor(zmain_stats['std'],  dtype=torch.float32)
            self.register_buffer('zmain_mean', zm_mean)
            self.register_buffer('zmain_inv_std', 1.0 / zm_std)
        # [NEW] REPA-E style: BN3d for z_main online stats learning.
        # train mode: batch stats normalize + running_stats EMA update.
        # eval mode: running stats normalize. ckpt 에 running stats 저장 → stage2 호환.
        self.normalize_zmain_bn = normalize_zmain_bn
        if normalize_zmain_bn:
            self.zmain_bn = nn.BatchNorm3d(
                z_dim, eps=1e-4, momentum=bn_momentum,
                affine=False,  # gamma/beta 없음 (z_prior 와 일관)
                track_running_stats=True,
            )
            if zmain_bn_init == 'zprior' and normalize_zprior:
                # REPA-E init_bn 과 동등: 사전 측정 stats 로 running_mean/var init
                with torch.no_grad():
                    self.zmain_bn.running_mean.copy_(self.prior_mean)
                    self.zmain_bn.running_var.copy_((1.0 / self.prior_inv_std).pow(2))
            elif zmain_bn_init == 'cold':
                with torch.no_grad():
                    self.zmain_bn.running_mean.zero_()
                    self.zmain_bn.running_var.fill_(1.0)
            # 'pytorch_default' = nn.BatchNorm3d 기본 init 그대로 (running_mean=0, var=1)
        # [NEW - oliviaa] RAE-style decoder noise augmentation
        # Per-sample random sigma in [0, tau], applied in normalized z space.
        # See: RAE paper (Diffusion Transformers with Representation Autoencoders).
        # Random mode per-sample: 0=main_only, 1=prior_only, 2=both
        # Curriculum: linearly ramp tau from 0 to final value over warmup_steps
        # (to avoid cold-start shock when resuming from clean-trained decoder).
        self.decoder_noise_tau_main = decoder_noise_tau_main
        self.decoder_noise_tau_prior = decoder_noise_tau_prior
        self.decoder_noise_random_mode = decoder_noise_random_mode
        self.decoder_noise_warmup_steps = decoder_noise_warmup_steps
        # power > 1: slow start, fast end (quadratic/cubic)
        # power = 1: linear
        # power < 1: fast start, slow end
        self.decoder_noise_warmup_power = decoder_noise_warmup_power
        # Counter for curriculum (per training forward call; auto-increments in self.training mode)
        self.register_buffer('_decoder_noise_step', torch.tensor(0, dtype=torch.long))
        self.align_weight = align_weight
        self.align_adaptive_weight = align_adaptive_weight
        # [NEW] 2-backward adaptive mode — mutually exclusive with _AdaptiveWeightingFn.
        # When True, forward() must NOT split z_cat; train loop calls
        # compute_adaptive_weight_2bwd() and scales align_loss explicitly.
        self.use_2backward_adaptive = use_2backward_adaptive
        self.adaptive_max_weight = adaptive_max_weight
        self.use_b_adaptive = use_b_adaptive  # [NEW - oliviaa/B-fix]

    def _norm_zprior(self, z_prior):
        if self.normalize_zprior:
            return (z_prior - self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)) * self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior)
        return z_prior

    def _denorm_zprior(self, z_prior):
        if self.normalize_zprior:
            return z_prior / self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior) + self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)
        return z_prior

    # [NEW - oliviaa] z_main normalize/denormalize (변종 B)
    def _norm_zmain(self, z_main):
        if self.normalize_zmain_bn:
            # REPA-E style BN: train mode 면 batch stats, eval mode 면 running stats
            return self.zmain_bn(z_main)
        if self.normalize_zmain:
            return (z_main - self.zmain_mean.view(1, -1, 1, 1, 1).to(z_main)) * self.zmain_inv_std.view(1, -1, 1, 1, 1).to(z_main)
        return z_main

    def _denorm_zmain(self, z_main):
        if self.normalize_zmain_bn:
            # BN 의 역변환: running stats 사용 (eval-style)
            mean = self.zmain_bn.running_mean.view(1, -1, 1, 1, 1).to(z_main)
            std = (self.zmain_bn.running_var + self.zmain_bn.eps).sqrt().view(1, -1, 1, 1, 1).to(z_main)
            return z_main * std + mean
        if self.normalize_zmain:
            return z_main / self.zmain_inv_std.view(1, -1, 1, 1, 1).to(z_main) + self.zmain_mean.view(1, -1, 1, 1, 1).to(z_main)
        return z_main

    def forward(self, x):
        # [NEW - oliviaa/B-fix] encode 의 return 가 tuple ((mu, log_var), (mu_adv, log_var_adv))
        # 가능 — use_b_adaptive=True + training 시.
        # 두 branch 모두 처리 (= rec, align 별도 graph node, same value).
        encode_result = self.vae.encode(x, scale=None)
        _is_b_adaptive_path = (isinstance(encode_result, tuple)
                                and len(encode_result) == 2
                                and isinstance(encode_result[0], tuple))
        if _is_b_adaptive_path:
            (mu, log_var), (mu_adv, log_var_adv) = encode_result
        else:
            mu, log_var = encode_result

        # reparameterize — same noise 두 branch 공유 (= 2backward 와 동등 비교 위함)
        if _is_b_adaptive_path:
            _std = torch.exp(0.5 * log_var)
            _noise = torch.randn_like(_std)
            z_main = _noise * _std + mu
            _std_adv = torch.exp(0.5 * log_var_adv)
            z_main_adv = _noise * _std_adv + mu_adv
        else:
            z_main = self.vae.reparameterize(mu, log_var)

        with torch.no_grad():
            z_prior = self.vae._encode_prior(x)
            z_prior = self._norm_zprior(z_prior)
        # [NEW - oliviaa] alignment 경로: z_main을 정규화 (변종 B)
        # decode 경로는 여전히 raw z_main 사용 (VAE 사전학습 호환)
        z_main_align = self._norm_zmain(z_main)
        z_cat = torch.cat([z_main_align, z_prior], dim=1)

        # Adaptive gradient weighting:
        # [NEW - oliviaa/B-fix] use_b_adaptive=True 시 — encode 가 이미 두 view 분기 →
        # z_cat_rec / z_cat 도 두 branch 의 별도 결과 사용. _AdaptiveWeightingFn.apply 사용 안 함.
        if self.use_b_adaptive and _is_b_adaptive_path:
            # adv branch 도 normalize + cat
            z_main_adv_align = self._norm_zmain(z_main_adv)
            z_cat_rec   = z_cat                                       # main = rec branch (= decoder)
            z_cat       = torch.cat([z_main_adv_align, z_prior], 1)   # adv = align branch
        elif (self.training and self.align_adaptive_weight and self.align_weight > 0
                and not self.use_2backward_adaptive):
            # 기존 single backward path (= activation gradient ratio)
            z_cat_rec, z_cat = _AdaptiveWeightingFn.apply(
                z_cat, z_cat.clone(), self.align_weight, 1e-6, self.adaptive_max_weight)
        else:
            z_cat_rec = z_cat

        # [NEW - oliviaa] RAE-style decoder noise augmentation
        # Per-sample uniform sigma ∈ [0, tau_eff], applied in NORMALIZED space.
        # tau_eff = tau * warmup_factor (curriculum: linear ramp 0 → tau over warmup_steps).
        # Random mode per-sample: 0=main_only, 1=prior_only, 2=both
        # Noise is applied to z_cat_rec (the decoder branch after the AW split).
        z_main_rec = z_cat_rec[:, :z_main.shape[1]]
        z_prior_rec = z_cat_rec[:, z_main.shape[1]:]
        if self.training and (self.decoder_noise_tau_main > 0 or self.decoder_noise_tau_prior > 0):
            B = z_main_rec.shape[0]
            device = z_main_rec.device
            dtype = z_main_rec.dtype

            # Curriculum warmup factor (power schedule: (t/T)^power)
            # power=1: linear, power=2: quadratic (slow start, fast end)
            if self.decoder_noise_warmup_steps > 0:
                cur = int(self._decoder_noise_step.item())
                progress = min(cur / float(self.decoder_noise_warmup_steps), 1.0)
                warmup_factor = progress ** self.decoder_noise_warmup_power
            else:
                warmup_factor = 1.0
            tau_main_eff = self.decoder_noise_tau_main * warmup_factor
            tau_prior_eff = self.decoder_noise_tau_prior * warmup_factor

            # Increment counter (only in training)
            self._decoder_noise_step += 1

            if self.decoder_noise_random_mode:
                # Per-sample random mode (1/4 each):
                #   0: clean (★no noise either, preserves clean recon ability★)
                #   1: main_only (z_main noise, z_prior clean)
                #   2: prior_only (z_prior noise, z_main clean)
                #   3: both noisy
                mode = torch.randint(0, 4, (B,), device=device)
                apply_main = ((mode == 1) | (mode == 3)).view(B, 1, 1, 1, 1).to(dtype)
                apply_prior = ((mode == 2) | (mode == 3)).view(B, 1, 1, 1, 1).to(dtype)
            else:
                apply_main = torch.ones((B, 1, 1, 1, 1), device=device, dtype=dtype)
                apply_prior = torch.ones((B, 1, 1, 1, 1), device=device, dtype=dtype)

            # z_main
            if tau_main_eff > 0 and self.normalize_zmain:
                sigma_m = tau_main_eff * torch.rand(
                    (B, 1, 1, 1, 1), device=device, dtype=dtype) * apply_main
                z_main_dec = self._denorm_zmain(z_main_rec + sigma_m * torch.randn_like(z_main_rec))
            else:
                # [fix - kk4aiq port] no-noise: decode 에 raw z_main 직접 사용.
                # 기존 _denorm_zmain(z_main_rec) 는 normalize(batch)→denorm(running) roundtrip 이라
                # train/eval BN stats 불일치 + running_var 오염 시 z_main 왜곡 → 제거.
                z_main_dec = z_main

            # z_prior
            if tau_prior_eff > 0 and self.normalize_zprior:
                sigma_p = tau_prior_eff * torch.rand(
                    (B, 1, 1, 1, 1), device=device, dtype=dtype) * apply_prior
                z_prior_dec = self._denorm_zprior(z_prior_rec + sigma_p * torch.randn_like(z_prior_rec))
            else:
                z_prior_dec = self._denorm_zprior(z_prior_rec)

            z_cat_raw = torch.cat([z_main_dec, z_prior_dec], dim=1)
        else:
            # [fix - kk4aiq port] decode 에 raw z_main 직접 사용 (denorm roundtrip 제거).
            # z_prior 는 고정 stats(prior_mean/inv_std) 라 _denorm_zprior 가 정확한 역변환 → 유지.
            z_cat_raw = torch.cat([z_main, self._denorm_zprior(z_prior_rec)], dim=1)
        recon = self.vae.decode(z_cat_raw, scale=None)
        return recon, mu, log_var, z_cat




class _AdaptiveWeightingFn(torch.autograd.Function):
    """Adaptive loss weighting via gradient norm equalization.

    Splits a shared tensor into two branches (x → loss_a, y → loss_b).
    In backward, scales grad_y so that its norm equals alpha * ||grad_x||,
    making loss_b contribute alpha times the gradient magnitude of loss_a.
    All-reduces norms across DDP ranks so every rank applies the same coefficient.
    """

    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                y: torch.Tensor,
                alpha: Union[float, torch.Tensor],
                eps: float,
                max_weight: float = 1e4) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.alpha = alpha
        ctx.eps = eps
        ctx.max_weight = max_weight
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        if grad_x is None or grad_y is None:
            return grad_x, grad_y, None, None, None

        work_dtype = torch.promote_types(grad_y.dtype, torch.float32)
        nx_sq = grad_x.detach().to(work_dtype).pow(2).sum()
        ny_sq = grad_y.detach().to(work_dtype).pow(2).sum()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nx_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(ny_sq, op=dist.ReduceOp.SUM)

        alpha = ctx.alpha
        if torch.is_tensor(alpha):
            alpha = alpha.detach().to(nx_sq.dtype)
        ratio = (nx_sq / ny_sq.clamp_min(ctx.eps)).sqrt()
        if ctx.max_weight > 0:
            ratio = ratio.clamp(0.0, ctx.max_weight)
        c = (alpha * ratio).to(grad_y.dtype)
        _AdaptiveWeightingFn._last_c = c.detach()

        return grad_x, c * grad_y, None, None, None


# ─── [LEGACY 2-backward adaptive] ───────────────────────────────
# Activated only by --use_2backward_adaptive. Mutually exclusive with the
# single-backward _AdaptiveWeightingFn path above: when 2-backward is on,
# GeopriorDiTAlignModel.forward must NOT call _AdaptiveWeightingFn.apply,
# and the train loop must call compute_adaptive_weight_2bwd here instead.
def compute_adaptive_weight_2bwd(rec_loss, align_loss, last_layer, max_weight=1e4, eps=1e-6):
    """Adaptive weight via two autograd.grad calls with retain_graph=True.

    w = ||d rec_loss / d last_layer|| / ||d align_loss / d last_layer||
    This is the original 2-backward path (pre _AdaptiveWeightingFn). Adds ~20-30%
    step time and FSDP2 reshard pressure due to retain_graph, but kept available
    for parity comparisons against the single-backward path.

    Args:
        rec_loss:    scalar loss whose grad is the reference scale
        align_loss:  scalar loss whose grad is scaled to match rec_loss
        last_layer:  parameter tensor (typically encoder.head[-1].weight)
        max_weight:  upper clamp for w (0 or negative disables clamp)
        eps:         numerical floor for align grad norm

    Returns:
        (w_clamped, w_raw): both detached scalar tensors
    """
    rec_grads   = torch.autograd.grad(rec_loss,   last_layer, retain_graph=True)[0]
    align_grads = torch.autograd.grad(align_loss, last_layer, retain_graph=True)[0]
    rec_norm = torch.norm(rec_grads)
    align_norm = torch.norm(align_grads)
    w = rec_norm / (align_norm + eps)
    w_raw = w.detach()
    if max_weight > 0:
        w = w.clamp(0.0, max_weight)
    # [DEBUG v23] norm 도 반환 (= (B) AW backward 의 grad_W_main_norm/adv_norm 과 비교)
    return w.detach(), w_raw, rec_norm.detach(), align_norm.detach()




# ─── Utilities (from train_causalvae.py) ─────────────────────

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        f"[rank{rank}] %(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        reset=True,
        style="%",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(stream_handler)

    return logger

def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)


def get_exp_name(args):
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}"


def set_train(modules):
    for module in modules:
        module.train()


def set_eval(modules):
    for module in modules:
        module.eval()


def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
    lora_state_dict={},
    optimizer_step=None,
    grad_accum_steps=None,
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_step": optimizer_step,  # accum 변경 시 wandb step 일관성 유지
            "grad_accum_steps": grad_accum_steps,  # save 시점의 accum (resume 시 fallback 계산용)
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "lora_state_dict": lora_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    # [NEW] keep_last rotation — 디스크 풀 방지 (save_checkpoint 는 rank0 전용).
    # CKPT_KEEP_LAST(기본 5)개의 최신 checkpoint-*.ckpt 만 유지, 나머지 삭제.
    # 무한 누적(8.6G/500step)으로 lustre 79T 꽉 차서 torch.save iostream 크래시 났던 것 방지.
    try:
        import glob as _glob, re as _re
        _keep = int(os.environ.get("CKPT_KEEP_LAST", "5"))
        if _keep > 0:
            def _stp(p):
                _m = _re.search(r"checkpoint-(\d+)\.ckpt", p)
                return int(_m.group(1)) if _m else -1
            _all = sorted(
                [c for c in _glob.glob(str(checkpoint_dir / "checkpoint-*.ckpt")) if _stp(c) >= 0],
                key=_stp,
            )
            for _old in _all[:-_keep]:
                try:
                    os.remove(_old)
                except OSError:
                    pass
    except Exception:
        pass
    return filepath


def valid(global_rank, rank, model, val_dataloader, precision, args, lpips_model=None):
    # [Modified - oliviaa] lpips_model을 외부에서 받아 재사용.
    # 기존에는 매 valid() 호출마다 AlexNet을 새로 할당했는데,
    # EMA 활성화 시 eval step당 2번 호출 → 두 번째 호출에서 OOM 발생.
    if args.eval_lpips and lpips_model is None:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    video_log = []
    num_video_log = args.eval_num_video_log

    # [NEW - oliviaa] CKNNA용 latent 수집 (dual_branch 모델에서만)
    # [MODIFIED - oliviaa/dit_align] GeopriorDiTAlignModel wrapper 경유
    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'vae'):
        raw_model = raw_model.vae
    is_dual_branch = getattr(raw_model, 'dual_branch', False)
    z_main_vecs = []   # list of (B, C) CPU tensors
    z_prior_vecs = []  # list of (B, C) CPU tensors
    z_cat_vecs = []    # [NEW - oliviaa/dit_align] z_cat (student latent) for alignment CKA
    z_ref_vecs = []    # [NEW - oliviaa/dit_align] z_ref (teacher latent) for alignment CKA

    # [NEW - jeeyoung] SSVAE-style diffusability — z_main(=encoder mu) per video 수집.
    #   각 batch 의 mu (B, C, T, H, W) 를 video 단위로 쪼개 CPU float 로 보관(GPU mem 절약).
    zmain_list = []

    # [NEW] decoder noise robustness — z_main 에 sigma*std 노이즈 후 decode, clean recon 대비 PSNR.
    _noise_sigmas = ([float(x) for x in getattr(args, 'noise_robust_sigmas', '0.1,0.2,0.3,0.5').split(',')]
                     if getattr(args, 'eval_noise_robust', False) else [])
    noise_robust_acc = {s: [] for s in _noise_sigmas}

    # [NEW - oliviaa/dit_align] dit_pipe 접근 — valid() 밖에서 주입
    _dit_pipe = getattr(valid, '_dit_pipe', None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank)
            # [NEW] 긴(>=33f) eval 은 chunk 버그 회피 위해 single-pass decode 강제 (256x17 등 짧은 건 chunk 그대로).
            #   decode 의 force_single_pass 플래그 (kinemadae_geoprior.py WanVAE_.decode) 를 frame 수로 토글.
            (model.module if hasattr(model, 'module') else model).vae.force_single_pass = (inputs.shape[2] >= 33)
            with torch.cuda.amp.autocast(dtype=precision):
                outputs = model(inputs)
                video_recon = outputs[0]

            # [NEW - jeeyoung] decode frame 수 검증용 — 첫 배치의 입력 T vs decode 출력 T 캡처.
            #   학습 중 chunked-decode frame 손실(17->15 / 81->71) 없는지 valid_model 에서 로깅.
            if batch_idx == 0:
                valid._frame_check = (int(inputs.shape[2]), int(video_recon.shape[2]))

            # [NEW - jeeyoung] diffusability 용 z_main(mu) 수집.
            #   encode 의 return 가 (mu, log_var) 또는 ((mu, log_var), (mu_adv, log_var_adv)).
            #   forward() (line 284-295) 와 동일한 tuple-handling 으로 첫 mu 만 취함.
            with torch.cuda.amp.autocast(dtype=precision):
                _enc = model.module.vae.encode(inputs, scale=None)
                _is_b_adaptive = (isinstance(_enc, tuple) and len(_enc) == 2
                                  and isinstance(_enc[0], tuple))
                if _is_b_adaptive:
                    (mu, _log_var), (_mu_adv, _log_var_adv) = _enc
                else:
                    mu, _log_var = _enc
            for b in range(mu.shape[0]):
                zmain_list.append(mu[b].detach().float().cpu())

            # Upload videos
            if global_rank == 0:
                for i in range(len(video_recon)):
                    if num_video_log <= 0:
                        break
                    # [FIX] tensor_to_video 는 [0,1] 입력 기대(내부에서 2x-1). 입력/recon 은 [-1,1] 이므로 [0,1] 매핑 후 전달 (안 하면 영상 어두워짐).
                    gt_video = tensor_to_video((inputs[i] + 1.0) / 2.0)
                    rec_video = tensor_to_video((torch.clamp(video_recon[i], -1.0, 1.0) + 1.0) / 2.0)
                    # [FIX - jeeyoung] gt/recon (T,C,H,W) 프레임·H 불일치 방어 — 공통 길이로 crop 후 concat(axis=3=W).
                    #   stage1 recon 은 같은 입력이라 보통 동일하지만 stage2/align gen-snap 버그와 일관되게 방어.
                    _tt = min(gt_video.shape[0], rec_video.shape[0]); _hh = min(gt_video.shape[2], rec_video.shape[2])
                    gt_video = gt_video[:_tt, :, :_hh]; rec_video = rec_video[:_tt, :, :_hh]
                    concat_video = np.concatenate([gt_video, rec_video], axis=3)
                    video_log.append(concat_video)
                    num_video_log -= 1
            # [NEW - oliviaa] CKNNA용 latent 수집
            if is_dual_branch:
                with torch.cuda.amp.autocast(dtype=precision):
                    z_main = outputs[1]  # mu: (B, z_dim, T', H', W')
                    z_prior = raw_model._encode_prior(inputs)
                    # [NEW - oliviaa/dit_align] z_cat = concat, z_ref = teacher VAE encode
                    if len(outputs) > 3:
                        z_cat = outputs[3]  # GeopriorDiTAlignModel returns 4-tuple
                    else:
                        z_cat = torch.cat([raw_model.reparameterize(outputs[1], outputs[2]), z_prior], dim=1)
                z_main_vecs.append(z_main.mean(dim=(2, 3, 4)).detach().cpu())
                z_prior_vecs.append(z_prior.mean(dim=(2, 3, 4)).detach().cpu())
                z_cat_vecs.append(z_cat.mean(dim=(2, 3, 4)).detach().cpu())
                # [NEW - oliviaa/dit_align] teacher VAE encode → z_ref
                if _dit_pipe is not None:
                    with torch.cuda.amp.autocast(dtype=precision):
                        z_ref_batch = []
                        for i in range(inputs.shape[0]):
                            z_i = _dit_pipe.vae.encode(
                                [inputs[i].to(dtype=precision)], device=rank, tiled=True
                            )[0]
                            z_ref_batch.append(z_i.mean(dim=(1, 2, 3)))  # global avg pool → (C,)
                        z_ref_vecs.append(torch.stack(z_ref_batch).detach().cpu())

            # [NEW] noise robustness — z_main 에 sigma*std 노이즈 추가 후 decode, "같은 base 의 clean decode" 대비 PSNR.
            #   [FIX 2026-06] 기존 버그: clean ref 로 video_recon(forward 의 reparameterize 샘플#1) 을 쓰고,
            #     _zcat 은 새 reparameterize(샘플#2) 로 만들어 비교 → forward 가 (x_recon, mu, log_var) 3개만 반환해
            #     len(outputs)>3 가 항상 False → 매번 새 샘플 → sigma=0 에서도 두 샘플 차이만큼 baseline mse →
            #     480x832x81 noise_robust 가 9.5 로 평평(baseline 이 added noise 를 압도)했음.
            #   수정: mu(outputs[1], deterministic) 로 z_main base 고정 + 같은 _zcat 의 clean decode 대비 비교
            #     → reparam randomness 제거 + sigma=0 baseline 0 보장. (clean decode 1회 추가, 출력 tensor 만 보관해 메모리 cheap.)
            if _noise_sigmas:
                _vae = model.module.vae
                _zdim = getattr(_vae, 'z_dim', 16)
                _zp = raw_model._encode_prior(inputs)
                _zcat = torch.cat([outputs[1], _zp], dim=1)  # mu(deterministic), reparameterize 아님
                # [FIX 2026-06-27] per-channel std — 디퓨전/flow-matching은 채널별 표준화((z-μ_c)/std_c) 후 unit noise라
                #   원본 공간 노이즈는 채널별 std_c 비례여야 함. 글로벌 std 1개는 채널 편차(측정 2.3배)만큼 어긋남
                #   (작은채널 과노이즈/큰채널 과소노이즈). diffusability_pr/lowfreq도 채널별 표준화라 일관성도 맞춤.
                _zstd = _zcat[:, :_zdim].float().std(dim=(0, 2, 3, 4), keepdim=True).to(_zcat.dtype)  # (1,C,1,1,1)
                with torch.cuda.amp.autocast(dtype=precision):
                    _clean_nr = _vae.decode(_zcat, scale=None)  # 같은 base 의 clean decode (baseline=0 보장)
                for _s in _noise_sigmas:
                    _zc = _zcat.clone()
                    _zc[:, :_zdim] = _zc[:, :_zdim] + (_s * _zstd) * torch.randn_like(_zc[:, :_zdim])
                    with torch.cuda.amp.autocast(dtype=precision):
                        _noisy = _vae.decode(_zc, scale=None)
                    _tt = min(_noisy.shape[2], _clean_nr.shape[2])
                    # [FIX] decode 출력 [-1,1] → [0,1] 매핑 후 MSE (PSNR과 동일 스케일, -6dB shift 제거)
                    _n01 = (torch.clamp(_noisy[:, :, :_tt].float(), -1.0, 1.0) + 1.0) / 2.0
                    _c01 = (torch.clamp(_clean_nr[:, :, :_tt].float(), -1.0, 1.0) + 1.0) / 2.0
                    _mse = torch.mean((_n01 - _c01) ** 2)
                    noise_robust_acc[_s].append((-10.0 * torch.log10(_mse + 1e-12)).item())
                del _clean_nr

            B, C, T, H, W = inputs.shape
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            video_recon = rearrange(
                video_recon, "b c t h w -> (b t) c h w"
            ).contiguous()

            # Calculate per-video PSNR (one value per video, not per batch)
            # to avoid partial-batch bias when DDP gather averages across ranks
            # [FIX] 입력이 [-1,1]이므로 PSNR은 [0,1]로 매핑 후 MAX=1 공식(범위 불변 = [0,1]-equivalent PSNR).
            #   매핑 안 하면 MAX=1에 [-1,1] 데이터라 -6dB 낮게 나옴. (LPIPS는 아래서 [-1,1] 원본 그대로 사용)
            _in01 = (inputs + 1.0) / 2.0
            _rec01 = (torch.clamp(video_recon, -1.0, 1.0) + 1.0) / 2.0
            mse = torch.mean(torch.square(_in01 - _rec01), dim=(1, 2, 3))  # (B*T,)
            psnr_frames = 20 * torch.log10(1 / torch.sqrt(mse))  # (B*T,)
            psnr_per_video = psnr_frames.view(B, T).mean(dim=1)   # (B,) mean over frames
            psnr_list.extend(psnr_per_video.detach().cpu().tolist())

            # Calculate per-video LPIPS
            if args.eval_lpips:
                lpips_frames = (
                    lpips_model.forward(inputs, video_recon)
                    .mean(dim=(1, 2, 3))  # (B*T,)
                )
                lpips_per_video = lpips_frames.view(B, T).mean(dim=1)  # (B,)
                lpips_list.extend(lpips_per_video.detach().cpu().tolist())

            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()

    # [NEW - jeeyoung] SSVAE diffusability metrics — per-channel STANDARDIZED z_main 위에서 계산.
    #   (1) few-mode participation ratio: correlation eigenspectrum 의 PR(=낮을수록 diffusable)
    #   (2) low-freq ratio: 3D-DCT power 의 low-freq corner(각 축 첫 1/4) 비중(=높을수록 diffusable).
    #   [변경 - jeeyoung] 기존엔 rank별 계산 후 mean → HD eval(영상 4개/8rank<4)서 NaN.
    #         이제 z_main 을 rank 간 all_gather 하여 rank0 에서 "전체 pool 1번" 계산 후 broadcast.
    #         (256 base 도 전체로 계산 → standalone 측정과 동일 방식. HD 도 gather 라 유효.)
    from scipy.fft import dctn   # np 는 모듈 레벨 사용 (함수 내 import 시 valid() 앞부분 np 사용이 UnboundLocalError)
    import torch.distributed as dist
    _ws = dist.get_world_size() if dist.is_initialized() else 1
    if _ws > 1:
        _gathered = [None for _ in range(_ws)]
        dist.all_gather_object(_gathered, [z.cpu() for z in zmain_list])   # 전 rank z_main(CPU) 수집
        all_zmain = [z for sub in _gathered if sub for z in sub] if rank == 0 else []
    else:
        all_zmain = zmain_list
    if rank == 0 or _ws == 1:
        if len(all_zmain) >= 4:
            Z = torch.stack(all_zmain).float()               # (Nv, C, T, H, W) — 전 rank pool
            Nv, Cc, T, H, W = Z.shape
            mean = Z.mean(dim=(0, 2, 3, 4), keepdim=True); std = Z.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
            Zs = (Z - mean) / std                            # per-channel unit-variance standardize
            X = Zs.permute(1, 0, 2, 3, 4).reshape(Cc, -1)
            cov = (X @ X.T) / (X.shape[1] - 1)
            ev = torch.linalg.eigvalsh(cov).clamp(min=0).flip(0)
            pr = (ev.sum() ** 2 / (ev ** 2).sum()).item()
            lf = []
            for v in range(Nv):
                D = dctn(Zs[v].numpy(), axes=(-3, -2, -1), norm='ortho'); P = (D ** 2).mean(0)
                lf.append(P[:max(1, T // 4), :max(1, H // 4), :max(1, W // 4)].sum() / P.sum())
            low_freq = float(np.mean(lf))
        else:
            pr, low_freq = float('nan'), float('nan')
    else:
        pr, low_freq = float('nan'), float('nan')

    # [변경 - jeeyoung] rank0 의 전체-pool 결과를 전 rank 로 broadcast (기존 all_reduce mean 대체)
    _t = torch.tensor([pr, low_freq], device=rank, dtype=torch.float32)
    if dist.is_initialized():
        dist.broadcast(_t, src=0)
    pr, low_freq = _t[0].item(), _t[1].item()

    # [NEW] noise robustness rank 평균 (sigma 별)
    noise_robust = {}
    if _noise_sigmas:
        _nr = torch.tensor([(np.mean(noise_robust_acc[s]) if noise_robust_acc[s] else float('nan'))
                            for s in _noise_sigmas], device=rank, dtype=torch.float32)
        _nr = torch.nan_to_num(_nr, nan=0.0)
        if dist.is_initialized():
            dist.all_reduce(_nr, op=dist.ReduceOp.SUM); _nr /= dist.get_world_size()
        noise_robust = {s: _nr[i].item() for i, s in enumerate(_noise_sigmas)}

    return psnr_list, lpips_list, video_log, z_main_vecs, z_prior_vecs, z_cat_vecs, z_ref_vecs, pr, low_freq, noise_robust


def gather_valid_result(psnr_list, lpips_list, video_log_list, rank, world_size,
                        z_main_vecs=None, z_prior_vecs=None):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_video_logs = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_video_logs, video_log_list)

    # [NEW - oliviaa] drift metrics (CKNNA, CKA, cosine sim): 각 rank의 z 벡터를 gather하여 rank 0에서 계산
    drift_metrics = None
    if z_main_vecs is not None and len(z_main_vecs) > 0:
        gathered_z_main = [None for _ in range(world_size)]
        gathered_z_prior = [None for _ in range(world_size)]
        z_main_cat = torch.cat(z_main_vecs, dim=0)   # (N_local, D)
        z_prior_cat = torch.cat(z_prior_vecs, dim=0)
        dist.all_gather_object(gathered_z_main, z_main_cat)
        dist.all_gather_object(gathered_z_prior, z_prior_cat)
        if rank == 0:
            all_z_main = torch.cat(gathered_z_main, dim=0)   # (N_total, D)
            all_z_prior = torch.cat(gathered_z_prior, dim=0)
            drift_metrics = {
                "cknna":      compute_cknna(all_z_main, all_z_prior, topk=10),
                "linear_cka": compute_linear_cka(all_z_main, all_z_prior),
                "cosine_sim": compute_mean_cosine_sim(all_z_main, all_z_prior),
            }

    return (
        np.mean(list(chain(*gathered_psnr_list))),
        np.mean(list(chain(*gathered_lpips_list))) if any(gathered_lpips_list) else 0.0,
        list(chain(*gathered_video_logs)),
        drift_metrics,
    )


# [NEW - oliviaa] alignment metrics — platonic-rep/metrics.py 기반으로 구현
# Source: /data/karlo-research_715/workspace/kinemadae/projects/oliviaa/platonic-rep/metrics.py

def _hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012, Eq.5)"""
    m = K.shape[0]
    K_t = K.clone().fill_diagonal_(0)
    L_t = L.clone().fill_diagonal_(0)
    hsic = (
        torch.sum(K_t * L_t.T)
        + torch.sum(K_t) * torch.sum(L_t) / ((m - 1) * (m - 2))
        - 2 * torch.sum(torch.mm(K_t, L_t)) / (m - 2)
    )
    return hsic / (m * (m - 3))


def _hsic_biased(K, L):
    """Biased HSIC via centering matrix H"""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    return torch.trace(K @ H @ L @ H)


def compute_cknna(z1, z2, topk=10):
    """
    CKNNA: kNN neighborhood에 HSIC 적용 (platonic-rep 기반)
    z1, z2: (N, D) CPU float tensors
    Returns score in [0, 1].
    """
    n = z1.shape[0]
    topk = min(topk, n - 1)
    if topk < 2:
        return 0.0
    # L2 normalize (platonic-rep test code 참고)
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    K = z1 @ z1.T  # cosine similarity matrix
    L = z2 @ z2.T
    # unbiased: 대각 -inf로 마스킹 후 topk
    K_hat = K.clone().fill_diagonal_(float('-inf'))
    L_hat = L.clone().fill_diagonal_(float('-inf'))
    _, idx_K = torch.topk(K_hat, topk, dim=1)
    _, idx_L = torch.topk(L_hat, topk, dim=1)
    mask_K = torch.zeros(n, n).scatter_(1, idx_K, 1.0)
    mask_L = torch.zeros(n, n).scatter_(1, idx_L, 1.0)
    # kNN intersection에 HSIC 적용
    sim_kl = _hsic_unbiased(mask_K * K, mask_L * L)
    sim_kk = _hsic_unbiased(mask_K * K, mask_K * K)
    sim_ll = _hsic_unbiased(mask_L * L, mask_L * L)
    return (sim_kl / (torch.sqrt(sim_kk * sim_ll) + 1e-6)).item()


def compute_linear_cka(z1, z2):
    """
    Linear CKA (platonic-rep 기반, biased HSIC)
    z1, z2: (N, D) CPU float tensors — D 달라도 됨
    Returns score in [0, 1].
    """
    z1 = z1.float()
    z2 = z2.float()
    K = z1 @ z1.T
    L = z2 @ z2.T
    hsic_kl = _hsic_biased(K, L)
    hsic_kk = _hsic_biased(K, K)
    hsic_ll = _hsic_biased(L, L)
    return (hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item()


# [NEW - oliviaa] Mean cosine similarity: per-sample 방향 유사도
# z_main(z_dim)과 z_prior(prior_z_dim)의 차원이 같을 때만 유효
def compute_mean_cosine_sim(z1, z2):
    """
    z1, z2: (N, D) CPU float tensors — D가 같아야 함
    Returns mean cosine similarity in [-1, 1], 또는 차원 불일치 시 None.
    """
    if z1.shape[1] != z2.shape[1]:
        return None  # z_dim != prior_z_dim인 경우 스킵
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    return (z1 * z2).sum(-1).mean().item()


def train(args):
    # Setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    logger = setup_logger(rank)

    # Init
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f"`{ckpt_dir}` exists!")
            time.sleep(5)
    dist.barrier()

    # [Modified - oliviaa] Load generator model
    # 원본: ModelRegistry.get_model() + from_pretrained/from_config로 OSP VAE 로드
    # 변경: _video_vae()로 Wan VAE 로드. ModelRegistry는 OSP 전용 레지스트리라 불필요.
    # add_stages는 JSON 문자열로 받아 파싱 (예: '[{"mode":"downsample3d","num_res_blocks":2}]')
    import json
    add_encoder_stages = json.loads(args.add_encoder_stages) if args.add_encoder_stages else None
    add_decoder_stages = json.loads(args.add_decoder_stages) if args.add_decoder_stages else None

    model = _video_vae(
        pretrained_path=args.pretrained_model_name_or_path,
        z_dim=args.z_dim,  # [Modified - oliviaa] 하드코딩 16 → argparse에서 받음
        device='cpu',
        add_encoder_stages=add_encoder_stages,
        add_decoder_stages=add_decoder_stages,
    )

    # [NEW - oliviaa] Freeze pretrained weights, keep added stages trainable
    # Wan pretrained에서 시작하므로 기존 weight는 고정하고 새 stage만 학습
    if args.freeze_pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.encoder.add_downsamples.parameters():
            param.requires_grad = True
        # [NEW - oliviaa] --unfreeze_encoder: encoder 전체 풀기 (pretrained 포함)
        if getattr(args, 'unfreeze_encoder', False):
            model.encoder.requires_grad_(True)
            if global_rank == 0:
                logger.warning("Encoder fully unfrozen (all pretrained encoder weights trainable).")
        # [NEW - oliviaa] --unfreeze_decoder: decoder 전체 풀기 (pretrained 포함)
        if getattr(args, 'unfreeze_decoder', False):
            model.decoder.requires_grad_(True)
            if global_rank == 0:
                logger.warning("Decoder fully unfrozen (all pretrained decoder weights trainable).")
        else:
            for param in model.decoder.add_upsamples.parameters():
                param.requires_grad = True
        # [NEW - oliviaa] z_dim이 pretrained(16)와 다르면 z_dim 관련 layer도 열어야 함
        # encoder head, conv1, conv2, decoder conv1이 z_dim에 의존
        if args.z_dim != 16:
            model.encoder.head[-1].requires_grad_(True)   # CausalConv3d(384→z_dim*2)
            model.conv1.requires_grad_(True)              # CausalConv3d(z_dim*2→z_dim*2)
            model.conv2.requires_grad_(True)              # CausalConv3d(z_dim→z_dim)
            model.decoder.conv1.requires_grad_(True)      # CausalConv3d(z_dim→384)
            if global_rank == 0:
                logger.warning(f"z_dim={args.z_dim} != pretrained(16). z_dim-related layers unfrozen.")
        # [NEW - oliviaa] --freeze_encoder: encoder 전체 freeze (decoder-only 학습용)
        if getattr(args, 'freeze_encoder', False):
            model.encoder.requires_grad_(False)
            model.conv1.requires_grad_(False)
            model.conv2.requires_grad_(False)
        # [NEW - oliviaa] --freeze_decoder: decoder 전체 freeze (Stage 1.5용)
        # student_patchify만 학습할 때 사용. VAE 전체가 frozen이라 rec_loss는 계산만 되고
        # gradient가 흐르지 않음 (불필요 compute지만 무시 가능).
        if getattr(args, 'freeze_decoder', False):
            model.decoder.requires_grad_(False)
            if global_rank == 0:
                logger.info("Decoder fully frozen (--freeze_decoder).")
            if global_rank == 0:
                logger.warning("Encoder fully frozen (decoder-only training mode).")
        if global_rank == 0:
            logger.warning("Pretrained weights frozen. Only added stages are trainable.")

    # [Modified - oliviaa] wandb 로깅 (TensorBoard 대체). working repo 와 동일 패턴.
    # --wandb_run_id 지정 시 resume, 없으면 새 run. WANDB_PROJECT env 로 project 설정.
    if global_rank == 0:
        logger.warning("Connecting to WANDB...")
        wandb_kwargs = dict(
            project=os.environ.get("WANDB_PROJECT", "kinemadae"),
            config=vars(args),
            name=get_exp_name(args),
        )
        if getattr(args, 'wandb_run_id', None):
            wandb_kwargs["id"] = args.wandb_run_id
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)

    dist.barrier()

    # [Modified - oliviaa] Load discriminator model
    # 원본: resolve_str_to_obj()로 문자열에서 클래스 찾기 — OSP 유틸 의존
    # 변경: LPIPSWithDiscriminator3D 직접 호출. resolve_str_to_obj 불필요.
    disc = LPIPSWithDiscriminator3D(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
        wavelet_weight=args.wavelet_weight
    )
    if getattr(args, 'lpips_chunk_size', 0) > 0:
        disc.lpips_chunk_size = args.lpips_chunk_size
        if global_rank == 0:
            logger.info(f"LPIPS chunk size: {args.lpips_chunk_size} (sequential computation for memory saving)")

    # ─── [NEW - oliviaa/dit_align] DiT pipeline + student patchify ───
    # DiT blocks 1 세트만 로드 (teacher/student 공유). patchify 만 별도.
    dit_pipe = None
    dit = None
    student_patchify = None
    if args.align_weight > 0:
        dit_pipe = load_pipeline(
            args.dit_ckpt_dir, f"cuda:{rank}",
            lora_checkpoint=args.lora_checkpoint if getattr(args, 'use_lora', False) else None,
            lora_target_modules=getattr(args, 'lora_target_modules', 'q,k,v,o,k_img,v_img,ffn.0,ffn.2'),
            lora_rank=getattr(args, 'lora_rank', 512),
        )
        dit = dit_pipe.dit  # WanModel
        dit.eval()
        # Freeze everything first; selectively unfreeze LoRA below.
        for p in dit.parameters():
            p.requires_grad = False

        # [NEW] LoRA: random-init inject when no ckpt + unfreeze LoRA params (student weights).
        if getattr(args, 'use_lora', False):
            if not args.lora_checkpoint:
                # load_pipeline only injects when lora_checkpoint is set → manually inject random init.
                from peft import LoraConfig, inject_adapter_in_model
                _target_modules = args.lora_target_modules.split(',')
                _lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                                          target_modules=_target_modules)
                dit = inject_adapter_in_model(_lora_config, dit)
                dit_pipe.dit = dit
                dit = dit.to(device=f"cuda:{rank}", dtype=torch.bfloat16)
                dit_pipe.dit = dit
            # Unfreeze only LoRA params (student-side trainable; pretrained DiT body frozen).
            # [FIX] keep LoRA params in bf16 (autocast/grad flow). bf16 학습에선 GradScaler 비활성.
            # 이전엔 fp32 cast 했지만, autocast(bf16) 안에서 fp32 weight 와 bf16 input mismatch 로
            # backward 시 LoRA params 의 .grad 가 None 이 됨 (확인된 버그).
            # [RESTORE] --freeze_lora: LoRA frozen(requires_grad=False) → optimizer 미포함. 1x52jvfx(loraFreeze) 재현.
            #   default(미지정)=trainable(기존동작). frozen 은 --use_lora 생략과 기능 동일(zero-init LoRA=0기여).
            _lora_trainable = not getattr(args, 'freeze_lora', False)
            _n_lora_params = 0
            for n, p in dit.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = _lora_trainable
                    _n_lora_params += p.numel() if _lora_trainable else 0
            if global_rank == 0:
                logger.info(f"[LoRA] inject={'ckpt' if args.lora_checkpoint else 'random'} "
                            f"rank={args.lora_rank} targets=[{args.lora_target_modules}] "
                            f"trainable params={_n_lora_params:,}")
            # [NEW] resume LoRA weight from ckpt (lora_checkpoint 와 별개로 자동 처리)
            if args.resume_from_checkpoint:
                _ckpt_for_lora = torch.load(args.resume_from_checkpoint, map_location='cpu')
                _lora_sd = _ckpt_for_lora.get('lora_state_dict', {})
                if _lora_sd:
                    _loaded = 0
                    for n, p in dit.named_parameters():
                        if n in _lora_sd:
                            p.data.copy_(_lora_sd[n].to(p.device, p.dtype))
                            _loaded += 1
                    if global_rank == 0:
                        logger.info(f"[LoRA] resumed from ckpt: {_loaded} tensors loaded")
                del _ckpt_for_lora

            # [NEW - oliviaa] baseline(256 finetuned) LoRA 로 init (fresh start 시, resume 과 별개).
            # teacher/student 공유 DiT 의 LoRA 시작점을 random 대신 256 적응본으로. base Wan 은 동일.
            if getattr(args, 'init_lora_safetensors', None):
                from safetensors.torch import load_file as _load_sf
                _init_lora = _load_sf(args.init_lora_safetensors)
                _dit_named = dict(dit.named_parameters())
                _loaded = 0; _missing = []
                for _k, _v in _init_lora.items():
                    if _k in _dit_named:
                        _dit_named[_k].data.copy_(_v.to(_dit_named[_k].device, _dit_named[_k].dtype))
                        _loaded += 1
                    else:
                        _missing.append(_k)
                if global_rank == 0:
                    logger.info(f"[LoRA init] baseline LoRA from {args.init_lora_safetensors}: "
                                f"{_loaded}/{len(_init_lora)} loaded, {len(_missing)} missing")
                    if _missing:
                        logger.warning(f"[LoRA init] missing 샘플: {_missing[:3]}")

    # Student patchify (FSDP 전에 생성 — pretrained weight 복사 필요)
    if args.align_weight > 0:
        _mask_ch = 12 if args.mask_mode == 'dual12' else 8
        _patchify_in_ch = (args.z_dim + _known.prior_z_dim) * 2 + _mask_ch  # noisy(48) + mask + image(48)
        student_patchify = create_student_patchify(dit, in_channels=_patchify_in_ch,
                                                    init_mode=args.patchify_init,
                                                    mask_init=args.patchify_mask_init,
                                                    mask_mode=args.mask_mode,
                                                    z_dim=args.z_dim,
                                                    prior_z_dim=_known.prior_z_dim)
        student_patchify = student_patchify.to(rank)

        # z_prior patchify weight 고정 (pretrained copy 유지, z_main만 학습)
        # [NEW - oliviaa] diffusion: --diffusion_unfreeze_zprior_patchify 켜면 freeze hook 끔
        # → z_prior patchify 가 diffusion (+ align) gradient 로 학습 (= stage2 rm57nmln 동일).
        _diff_unfreeze_zprior = getattr(args, 'diffusion_unfreeze_zprior_patchify', False)
        if getattr(args, 'freeze_patchify_zprior', False) and not _diff_unfreeze_zprior:
            _z_dim = args.z_dim
            _prior_z_dim = _known.prior_z_dim
            def _zero_zprior_grad(grad):
                # z_prior: noisy(32:48), image_z_prior(마지막 16ch)
                grad[:, _z_dim:_z_dim+_prior_z_dim] = 0
                grad[:, -_prior_z_dim:] = 0
                return grad
            student_patchify.weight.register_hook(_zero_zprior_grad)
            if global_rank == 0:
                logger.info(f"Patchify z_prior channels frozen (weight[:, {_z_dim}:{_z_dim+_prior_z_dim}] + weight[:, -{_prior_z_dim}:])")
        elif _diff_unfreeze_zprior and global_rank == 0:
            logger.info("[diffusion] z_prior patchify UNFROZEN (freeze hook 생략, align+diffusion 학습)")
    else:
        if global_rank == 0:
            logger.info("align_weight=0: DiT and student_patchify not loaded (pure VAE training)")

    # [NEW] DiT 메모리 최적화
    _dit_offload, _dit_fsdp2, _align_block_set = setup_dit_memory(
        dit, args, rank, global_rank,
        has_fsdp2=HAS_FSDP2, fully_shard=fully_shard, MixedPrecisionPolicy=MixedPrecisionPolicy,
        logger=logger if global_rank == 0 else None,
    )

    # FlowMatchScheduler — align noise 추가용 (training=False, dit_num_inference_steps timestep)
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(args.dit_num_inference_steps)

    # [RESTORE matchS2] align_match_stage2 시 align noise 를 stage2 와 동일하게 — 1000-step training scheduler.
    #   training=True → linear_timesteps_weights(=bsmntw) 생성 → training_weight(timestep) per-sample weight.
    align_match_scheduler = None
    if getattr(args, 'align_match_stage2', False):
        align_match_scheduler = FlowMatchScheduler(template="Wan")
        align_match_scheduler.set_timesteps(1000, training=True)
        if global_rank == 0:
            logger.info(f"[matchS2] align_match_scheduler set (1000 steps, training=True, "
                        f"timesteps={len(align_match_scheduler.timesteps)}, "
                        f"has_weights={hasattr(align_match_scheduler, 'linear_timesteps_weights')}, "
                        f"boundary=[{args.align_min_timestep_boundary},{args.align_max_timestep_boundary}])")

    # [NEW - oliviaa] diffusion 전용 scheduler — stage2 (rm57nmln) 와 동일 (1000 step + training=True).
    # training=True → linear_timesteps_weights 생성 (= flow match training weight). align scheduler 와 분리.
    diffusion_scheduler = None
    if getattr(args, 'use_diffusion_loss', False):
        diffusion_scheduler = FlowMatchScheduler(template="Wan")
        diffusion_scheduler.set_timesteps(1000, training=True)
        if global_rank == 0:
            logger.info("[diffusion] diffusion_scheduler set (1000 steps, training=True, "
                        f"timesteps={len(diffusion_scheduler.timesteps)}, "
                        f"has_weights={hasattr(diffusion_scheduler, 'linear_timesteps_weights')})")

    # Text context: null prompt / per-batch caption / T5 cache
    _use_caption = getattr(args, 'caption_metadata', None) is not None
    _use_t5_cache = getattr(args, 't5_cache_dir', None) is not None
    caption_map = {}
    t5_cache = {}
    if _use_t5_cache:
        import glob as _glob_cache
        cache_files = sorted(_glob_cache.glob(os.path.join(args.t5_cache_dir, "shard_*.pt")))
        for cf in cache_files:
            shard = torch.load(cf, map_location='cpu', weights_only=False)
            t5_cache.update(shard)
            del shard
        if global_rank == 0:
            logger.info(f"Loaded {len(t5_cache)} T5 cached embeddings from {args.t5_cache_dir}")
            logger.info("T5 can be offloaded (using cache)")
    elif _use_caption:
        import json as _json_cap
        with open(args.caption_metadata) as f:
            for line in f:
                entry = _json_cap.loads(line)
                caption_map[entry['video']] = entry['prompt']
        if global_rank == 0:
            logger.info(f"Loaded {len(caption_map)} captions from {args.caption_metadata}")
            logger.info("T5 stays on GPU (per-batch encoding)")

    null_context = None
    if dit_pipe is not None:
        with torch.no_grad():
            null_context = prepare_null_context(dit_pipe)
        # T5 offload — caption mode에서는 T5가 매 배치 필요하므로 offload 불가
        # text_fsdp2와 상호 배타적: FSDP2 sharding이 활성화된 경우 offload 불필요
        _t5_offload_active = (getattr(args, 't5_offload', False) and not _use_caption or _use_t5_cache)
        if _t5_offload_active and getattr(args, 'text_fsdp2', False):
            if global_rank == 0:
                logger.warning("--t5_offload and --text_fsdp2 both set; text_fsdp2 takes precedence, skipping offload")
            _t5_offload_active = False
        if _t5_offload_active:
            if dit_pipe.text_encoder is not None:
                dit_pipe.text_encoder.to('cpu')
            torch.cuda.empty_cache()
            if global_rank == 0:
                logger.info("T5 offloaded to CPU. freed ~10GB VRAM")
        setup_text_encoder_memory(
            dit_pipe, args, rank, global_rank,
            has_fsdp2=HAS_FSDP2, fully_shard=fully_shard, MixedPrecisionPolicy=MixedPrecisionPolicy,
            logger=logger if global_rank == 0 else None,
        )
        if global_rank == 0:
            logger.info(f"DiT loaded. student_patchify in_channels={student_patchify.in_channels}")
            logger.info(f"DiT blocks: {len(dit.blocks)}, dim={dit.dim}")

    # [NEW - oliviaa] Stage 1.5: zmain_stats 로드 (--normalize_zmain 시에만 사용)
    _zmain_stats = None
    if getattr(args, 'normalize_zmain', False):
        if not getattr(args, 'zmain_stats_path', None):
            raise ValueError("--normalize_zmain requires --zmain_stats_path")
        import json as _json_zm
        with open(args.zmain_stats_path) as _f_zm:
            _zmain_stats = _json_zm.load(_f_zm)
        if global_rank == 0:
            logger.info(f"[Stage 1.5] Loaded zmain_stats from {args.zmain_stats_path} "
                        f"(z_dim={len(_zmain_stats.get('mean', []))})")

    # GeopriorDiTAlignModel wrapper (VAE only — student_patchify 는 DDP 밖)
    # [FIX - oliviaa] student_patchify 를 DDP wrapper 안에 넣으면 forward() 밖에서
    # 사용하는 param 이 DDP gradient hook 에서 "marked ready twice" 에러 발생.
    # student_patchify 는 별도로 관리하고 backward 후 수동 all_reduce.
    model = GeopriorDiTAlignModel(
        model,
        normalize_zprior=getattr(args, 'normalize_zprior', False),
        zmain_stats=_zmain_stats,
        decoder_noise_tau_main=getattr(args, 'decoder_noise_tau_main', 0.0),
        decoder_noise_tau_prior=getattr(args, 'decoder_noise_tau_prior', 0.0),
        decoder_noise_random_mode=getattr(args, 'decoder_noise_random_mode', True),
        decoder_noise_warmup_steps=getattr(args, 'decoder_noise_warmup_steps', 0),
        decoder_noise_warmup_power=getattr(args, 'decoder_noise_warmup_power', 1.0),
        align_weight=args.align_weight,
        align_adaptive_weight=args.align_adaptive_weight,
        use_2backward_adaptive=getattr(args, 'use_2backward_adaptive', False),
        adaptive_max_weight=getattr(args, 'adaptive_max_weight', 1e4),
        use_b_adaptive=getattr(args, 'use_b_adaptive', False),  # [NEW - oliviaa/B-fix]
        normalize_zmain_bn=getattr(args, 'normalize_zmain_bn', False),
        bn_momentum=getattr(args, 'bn_momentum', 0.1),
        zmain_bn_init=getattr(args, 'zmain_bn_init', 'zprior'),
        z_dim=args.z_dim,
        # [NEW - oliviaa] block 별 align projection
        use_align_projection=getattr(args, 'use_align_projection', False),
        align_proj_dim=getattr(args, 'align_proj_dim', 5120),
        align_proj_num_blocks=getattr(args, 'align_num_blocks', 40),
        align_projection_init=getattr(args, 'align_projection_init', 'zero'),
        align_proj_bottleneck_dim=getattr(args, 'align_proj_bottleneck_dim', 64),
    )
    # [NEW - oliviaa] diffusion loss 용 DaVaeHead 등록 (= stage2 rm57nmln 동일).
    # dit.head 복사 init (head_main zero, head_prior copy). DDP wrap 전 attach → grad sync.
    # align_weight>0 조건 추가: diffusion 은 align 블록 안에서만 돌므로, align_weight=0 인데
    # davae_head 등록되면 DDP static_graph 가 unused param 으로 에러 가능.
    if getattr(args, 'use_diffusion_loss', False) and dit is not None and args.align_weight > 0:
        from dit_align import build_davae_head
        _prior_z_dim = _known.prior_z_dim
        model.davae_head = build_davae_head(dit, z_dim=args.z_dim, prior_z_dim=_prior_z_dim)
        if global_rank == 0:
            _n = sum(p.numel() for p in model.davae_head.parameters())
            logger.info(f"[diffusion] DaVaeHead registered (z_main={args.z_dim}ch zero-init, "
                        f"z_prior={_prior_z_dim}ch copy-init, {_n:,} params)")
    else:
        model.davae_head = None

    # [NEW] SyncBN convert (multi-GPU 면) — REPA-E 따라
    if getattr(args, 'normalize_zmain_bn', False) and dist.get_world_size() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if global_rank == 0:
            logger.info(f"[zmain_bn] SyncBatchNorm applied (world_size={dist.get_world_size()})")

    model = model.to(rank)
    # [NEW - oliviaa] Stage 1.5: VAE 전체가 frozen이면 DDP wrap이 실패함
    # ("DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient")
    # → trainable param 없으면 DDP 건너뛰고 plain model 사용. .module 인터페이스는 wrapper로 유지.
    _has_trainable_vae = any(p.requires_grad for p in model.parameters())
    if _has_trainable_vae:
        # [NEW] static_graph=True: AdaptiveWeightedConv3d 의 (y, y) dual output + align_projections 의 dual-edge
        # backward 시 DDP 가 같은 param 의 grad hook 두 번 발동 → "marked as ready twice" 에러.
        # static_graph=True 는 graph 가 매 step 동일하다고 가정 → reducer 가 두 번 마킹 허용 (= 의도된 design).
        # find_unused_parameters=True 와 함께 사용 가능 (PyTorch docs 공식 지원).
        model = DDP(
            model, device_ids=[rank],
            find_unused_parameters=args.find_unused_parameters,
            static_graph=True,
        )
    else:
        if global_rank == 0:
            logger.info("VAE has no trainable params → skipping DDP wrap (Stage 1.5 patchify-only mode)")
        # Wrapper that mimics DDP interface (.module attribute + forward delegation)
        class _NoDDPWrapper:
            def __init__(self, m): object.__setattr__(self, 'module', m)
            def __call__(self, *args, **kwargs): return self.module(*args, **kwargs)
            def __getattr__(self, name):
                # fallback: delegate to underlying module
                return getattr(self.module, name)
            def no_sync(self): import contextlib; return contextlib.nullcontext()
            def train(self, mode=True): self.module.train(mode); return self
            def eval(self): self.module.eval(); return self
            def to(self, *a, **k): self.module.to(*a, **k); return self
        model = _NoDDPWrapper(model)
    disc = disc.to(rank)
    disc = DDP(
        disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )

    # Load dataset
    dataset = TrainVideoDataset(
        args.video_path,
        sequence_length=args.num_frames,
        resolution=args.resolution,
        sample_rate=args.sample_rate,
        dynamic_sample=args.dynamic_sample,
        cache_file="idx.pkl",
        is_main_process=global_rank == 0,
    )
    ddp_sampler = CustomDistributedSampler(dataset)
    if args.mix_frames:
        # [NEW] 17/81 혼합 — batch_sampler 는 batch_size/sampler/shuffle 와 동시 사용 불가
        batch_sampler = MixedLengthBatchSampler(
            ddp_sampler,
            batch_size=args.batch_size,
            base_num_frames=args.num_frames,
            mix_81_prob=args.mix_81_prob,
            mix_81_num_frames=args.mix_81_num_frames,
            mix_81_batch_size=args.mix_81_batch_size,
            seed=getattr(args, "seed", 0),
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,        # batch_size/sampler 없이
            pin_memory=True,
            num_workers=args.dataset_num_worker,
        )
    else:
        dataloader = DataLoader(                # 기존 경로 (그대로)
            dataset,
            batch_size=args.batch_size,
            sampler=ddp_sampler,
            pin_memory=True,
            num_workers=args.dataset_num_worker,
        )
    val_dataloader = None
    if args.eval_video_path is not None:
        val_dataset = ValidVideoDataset(
            real_video_dir=args.eval_video_path,
            num_frames=args.eval_num_frames,
            sample_rate=args.eval_sample_rate,
            crop_size=args.eval_resolution,
            resolution=args.eval_resolution,
        )
        indices = range(args.eval_subset_size)
        val_dataset = Subset(val_dataset, indices=indices)
        val_sampler = CustomDistributedSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            pin_memory=True,
        )

    # [NEW - oliviaa] Additional eval dataloaders for higher resolutions
    # --eval_resolutions_hd: comma-separated list e.g. "512x512,480x832"
    hd_val_dataloaders = []  # list of (name_tag, dataloader)
    if getattr(args, 'eval_resolutions_hd', None):
        for res_str in args.eval_resolutions_hd.split(','):
            res_str = res_str.strip()
            hd_num_frames = args.eval_num_frames           # [NEW] default = 기존 eval frame 수
            if 'x' in res_str:
                parts = list(map(int, res_str.split('x')))
                if len(parts) == 3:                        # [NEW] "HxWxF" → frame 수 지정 (예: 480x832x81)
                    h, w, hd_num_frames = parts
                    name_tag = f"{h}x{w}x{hd_num_frames}"
                else:
                    h, w = parts
                    name_tag = f"{h}x{w}"
                res = (h, w)
            else:
                res = int(res_str)
                name_tag = str(res)
            hd_bs = max(1, args.eval_batch_size // 4)  # 고해상도는 batch 줄임
            hd_dataset = ValidVideoDataset(
                real_video_dir=args.eval_video_path,
                num_frames=hd_num_frames,                  # [NEW] HxWxF면 F, 아니면 eval_num_frames
                sample_rate=args.eval_sample_rate,
                crop_size=res,
                resolution=res,
            )
            hd_subset = Subset(hd_dataset, indices=range(args.eval_hd_subset_size))  # [NEW] base eval과 분리
            hd_sampler = CustomDistributedSampler(hd_subset)
            hd_loader = DataLoader(hd_subset, batch_size=hd_bs, sampler=hd_sampler, pin_memory=True)
            hd_val_dataloaders.append((name_tag, hd_loader))

    # [MODIFIED - oliviaa/dit_align] Optimizer — VAE + patchify 별도 param group.
    # model.module = GeopriorDiTAlignModel(vae, student_patchify)
    vae_module = model.module.vae
    patchify_module = student_patchify  # DDP 밖에서 별도 관리 (None if no align)

    vae_params = [p for p in vae_module.parameters() if p.requires_grad]
    if patchify_module is not None and getattr(args, 'freeze_patchify_full', False):
        for _p in patchify_module.parameters():
            _p.requires_grad = False
        patchify_params = []
        if global_rank == 0:
            logger.info("student_patchify fully frozen (--freeze_patchify_full).")
    else:
        patchify_params = list(patchify_module.parameters()) if patchify_module is not None else []

    # modules_to_train: set_train/set_eval 에서 사용
    if getattr(args, 'unfreeze_encoder', False):
        enc_module = vae_module.encoder
    else:
        enc_module = vae_module.encoder.add_downsamples
    if getattr(args, 'unfreeze_decoder', False):
        modules_to_train = [enc_module, vae_module.decoder]
    else:
        modules_to_train = [enc_module, vae_module.decoder.add_upsamples]
    if patchify_module is not None and not getattr(args, 'freeze_patchify_full', False):
        modules_to_train.append(patchify_module)
    if getattr(args, 'expand_encoder_head', False):
        modules_to_train += [vae_module.encoder.head, vae_module.conv1]
    if getattr(model.module, 'align_projections', None) is not None:
        modules_to_train.append(model.module.align_projections)
    # [NEW - oliviaa] diffusion: davae_head 학습 대상 추가
    _davae_head = getattr(model.module, 'davae_head', None)
    if _davae_head is not None:
        modules_to_train.append(_davae_head)
    # [NEW - oliviaa] diffusion: block norms + modulation unfreeze (= rm57nmln unfreeze_block_norms_mod)
    _block_norms_mod_params = []
    if getattr(args, 'diffusion_unfreeze_block_norms_mod', False) and dit is not None and hasattr(dit, 'blocks'):
        _bn_paths = ['norm3', 'self_attn.norm_q', 'self_attn.norm_k',
                     'cross_attn.norm_q', 'cross_attn.norm_k', 'cross_attn.norm_k_img']
        for blk in dit.blocks:
            if hasattr(blk, 'modulation'):
                blk.modulation.requires_grad = True
                _block_norms_mod_params.append(blk.modulation)
            for _path in _bn_paths:
                _mod = blk
                try:
                    for _part in _path.split('.'):
                        _mod = getattr(_mod, _part)
                    for _pp in _mod.parameters():
                        _pp.requires_grad = True
                        _block_norms_mod_params.append(_pp)
                except AttributeError:
                    pass
        if global_rank == 0:
            logger.info(f"[diffusion] block norms+mod unfrozen: {len(_block_norms_mod_params)} tensors "
                        f"({sum(p.numel() for p in _block_norms_mod_params):,} params)")

    # [NEW] teacher_frozen_pretrained: teacher forward(align target)를 init Wan I2V 로 고정.
    #   block_norms/mod 초기값(pretrained, 학습 전)을 snapshot → teacher forward 시 이 값으로 swap.
    #   LoRA 는 enable_adapters(False) 로 끄면 되니 snapshot 불필요 (다시 켜면 학습된 LoRA 복원됨).
    _teacher_frozen_pretrained = getattr(args, 'teacher_frozen_pretrained', False)
    _teacher_bn0 = {}
    if _teacher_frozen_pretrained:
        from peft.tuners.tuners_utils import BaseTunerLayer  # teacher forward 에서 LoRA 토글용
        for _p in _block_norms_mod_params:
            _teacher_bn0[id(_p)] = _p.detach().clone()
        if global_rank == 0:
            logger.info(f"[teacher_frozen] align target = init pretrained Wan I2V. "
                        f"block_norms snapshot: {len(_teacher_bn0)} tensors "
                        f"({sum(p.numel() for p in _block_norms_mod_params):,} params). "
                        f"teacher forward 시 LoRA off + 이 값으로 swap → 직후 원복")

    param_groups = [{'params': vae_params, 'lr': args.lr}]
    if patchify_params:
        param_groups.append({'params': patchify_params, 'lr': args.patchify_lr})
    # [NEW] LoRA params on student DiT (trainable). Same lr as patchify_lr.
    if getattr(args, 'use_lora', False) and dit is not None:
        lora_params = [p for n, p in dit.named_parameters() if 'lora_' in n and p.requires_grad]
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': args.patchify_lr})
            if global_rank == 0:
                logger.info(f"[LoRA] optimizer received {len(lora_params)} param tensors "
                            f"({sum(p.numel() for p in lora_params):,} elements) at lr={args.patchify_lr}")
    if getattr(model.module, 'align_projections', None) is not None:
        align_proj_params = [p for p in model.module.align_projections.parameters() if p.requires_grad]
        if align_proj_params:
            param_groups.append({'params': align_proj_params, 'lr': args.patchify_lr})
            if global_rank == 0:
                logger.info(f"[align_proj] optimizer received {len(align_proj_params)} param tensors "
                            f"({sum(p.numel() for p in align_proj_params):,} elements) at lr={args.patchify_lr}")
    # [NEW - oliviaa] diffusion: davae_head + block_norms_mod 를 optimizer 에 추가
    if _davae_head is not None:
        davae_params = [p for p in _davae_head.parameters() if p.requires_grad]
        if davae_params:
            param_groups.append({'params': davae_params, 'lr': args.patchify_lr})
            if global_rank == 0:
                logger.info(f"[diffusion] davae_head optimizer received {len(davae_params)} tensors "
                            f"({sum(p.numel() for p in davae_params):,} params) at lr={args.patchify_lr}")
    if _block_norms_mod_params:
        param_groups.append({'params': _block_norms_mod_params, 'lr': args.patchify_lr})
        if global_rank == 0:
            logger.info(f"[diffusion] block_norms_mod optimizer received {len(_block_norms_mod_params)} tensors "
                        f"({sum(p.numel() for p in _block_norms_mod_params):,} params) at lr={args.patchify_lr}")
    gen_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    _warmup_target_lrs = [pg['lr'] for pg in gen_optimizer.param_groups]  # [NEW] warmup: per-group 목표 lr 보존 (ramp 끝나면 이 값)
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=args.lr, weight_decay=0.01
    )

    # AMP scaler — disabled when LoRA + bf16 (mixed Tensor/DTensor unscale_ fails).
    # bf16 has its own numerical range so GradScaler is unnecessary anyway.
    _scaler_enabled = not (getattr(args, 'use_lora', False) and args.mix_precision == 'bf16')
    scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)
    disc_scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)  # [NEW] disc 용 별도 scaler (accum과 충돌 방지)
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    print(precision)
    
    # Load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)
        # [FIX - oliviaa] decoder_only ckpt는 student_patchify가 빈 dict {}로 저장돼 있음 (align_weight=0).
        # 빈 dict로 strict load 시 "Missing keys" 에러 → 비어있으면 skip하고 fresh init 사용.
        _stu_pe_state = checkpoint["state_dict"].get("student_patchify")
        if _stu_pe_state and student_patchify is not None:
            student_patchify.load_state_dict(_stu_pe_state)
            logger.info(f"Loaded student_patchify from resume ckpt ({len(_stu_pe_state)} keys)")
        elif student_patchify is not None:
            logger.info("student_patchify state empty in resume ckpt → using fresh init")
        disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        # [NEW - oliviaa] trainable params가 바뀐 경우 optimizer state 크기 불일치 → skip
        try:
            gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
            disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Optimizer state 로드 스킵 (trainable params 변경으로 인한 불일치): {e}")
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        # accum 변경 시 wandb step 일관성 유지:
        # 1) ckpt 에 optimizer_step 있으면 그대로 복원
        # 2) 없으면 ckpt 의 grad_accum_steps 로 계산 (= current_step // ckpt_accum)
        # 3) 둘 다 없으면 학습 시점의 args.grad_accum_steps 로 fallback
        _ckpt_optim_step = checkpoint.get("optimizer_step", None)
        # fallback 우선순위: ckpt 의 grad_accum_steps → args.resume_ckpt_grad_accum → args.grad_accum_steps
        _ckpt_accum = (checkpoint.get("grad_accum_steps", None)
                       or getattr(args, 'resume_ckpt_grad_accum', None)
                       or args.grad_accum_steps)
        _resume_optim_step = _ckpt_optim_step if _ckpt_optim_step is not None else (current_step // _ckpt_accum)
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step} (optim_step={_resume_optim_step})"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()
        if args.resume_from_checkpoint and checkpoint.get("ema_state_dict"):
            _ema_sd = checkpoint["ema_state_dict"]
            # [NEW] new format: {'shadow': ..., 'shadow_buffers': ...} / legacy: dict 자체가 shadow
            if isinstance(_ema_sd, dict) and 'shadow' in _ema_sd and 'shadow_buffers' in _ema_sd:
                _shadow_sd = _ema_sd['shadow']
                _shadow_buf_sd = _ema_sd['shadow_buffers']
            else:
                _shadow_sd = _ema_sd
                _shadow_buf_sd = {}
            # parameter shadow
            for name, param in model.named_parameters():
                if name in _shadow_sd:
                    ema.shadow[name] = _shadow_sd[name].to(dtype=param.dtype, device=param.device)
            # [NEW] buffer shadow (BN running stats EMA 등) — REPA-E style
            for name, buf in model.named_buffers():
                if name in _shadow_buf_sd:
                    ema.shadow_buffers[name] = _shadow_buf_sd[name].to(dtype=buf.dtype, device=buf.device)
            logger.info(f"EMA state loaded from checkpoint ({len(ema.shadow)} params, {len(ema.shadow_buffers)} buffers)")

    # [NEW - oliviaa] GAN adaptive weight용 last_layer 결정
    # [MODIFIED - oliviaa/dit_align] model.module = GeopriorDiTAlignModel → .vae 경유
    if args.gan_last_layer == "add_upsamples":
        gan_last_layer = model.module.vae.decoder.add_upsamples[-1][-1].resample[-1].weight
        logger.info(f"GAN last_layer: decoder.add_upsamples[-1][-1].resample[-1] (trainable)")
    else:
        gan_last_layer = model.module.vae.decoder.head[-1].weight
        logger.info(f"GAN last_layer: decoder.head[-1] (default)")


    # Training loop
    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        logger.info(f"=== Model Params ===")
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.vae.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.vae.decoder):d}M")
        if student_patchify is not None:
            logger.info(f"\t- Patchify:\t{total_params(student_patchify):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"===========")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")

    # Training Bar
    bar_desc = ""
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        bar_desc = "E{current_epoch} gen:{gen_loss} disc:{disc_loss} rec:{rec_loss} nll:{nll_loss} kl:{kl_loss} std:{latents_std}"
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    # Training Loop
    num_epochs = args.epochs

    # [Modified - oliviaa] progress bar에 주요 메트릭 표시
    last_metrics = {"gen_loss": "-", "disc_loss": "-", "rec_loss": "-", "nll_loss": "-", "kl_loss": "-", "latents_std": "-"}

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, **last_metrics)
            bar.update()

    # [NEW - oliviaa] LPIPS 모델을 학습 루프 전에 한 번만 생성.
    # 기존에는 valid() 내부에서 매번 생성 → EMA 있을 때 eval step당 2회 할당 → OOM.
    shared_lpips_model = None
    if args.eval_lpips:
        shared_lpips_model = lpips.LPIPS(net="alex", spatial=True)
        shared_lpips_model.to(rank)
        shared_lpips_model = DDP(shared_lpips_model, device_ids=[rank])
        shared_lpips_model.requires_grad_(False)
        shared_lpips_model.eval()

    # [NEW] grad accumulation 용 변수 초기화
    _accum = getattr(args, 'grad_accum_steps', 1)
    _loss_accum = {"g_loss": 0.0, "rec_loss": 0.0, "kl_loss": 0.0, "nll_loss": 0.0,
                   "align_loss": 0.0, "total_loss": 0.0}
    # [FIX] optim step 초기화: resume 시 ckpt 의 optim_step 사용 (= accum 변경 시 일관성). fresh 시 0.
    optimizer_step = _resume_optim_step if args.resume_from_checkpoint and '_resume_optim_step' in dir() else 0
    _warmup_anchor_step = optimizer_step  # [NEW] warmup 기준점 = 이 run 시작 optimizer_step (resume 후 N step 동안 0->target ramp)
    _initial_eval_done = False   # [NEW] resume/start 직후 1회 initial eval (8500 baseline 을 wandb step 처음에)

    if global_rank == 0:
        torch.cuda.empty_cache()
        _mem_train_start = torch.cuda.memory_allocated(rank) / 1e9
        _mem_train_reserved = torch.cuda.memory_reserved(rank) / 1e9
        logger.info(f"GPU {rank} memory at training start: allocated={_mem_train_start:.2f}GB, reserved={_mem_train_reserved:.2f}GB")

    for epoch in range(num_epochs):
        gen_optimizer.zero_grad()
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        for batch_idx, batch in enumerate(dataloader):
            # [FIX - oliviaa] max_steps 도달 시 조기 종료
            if args.max_steps is not None and current_step >= args.max_steps:
                break
            inputs = batch["video"].to(rank)
            # [NEW - mixed-length] 이번 배치 프레임수 로깅 (17/81 섞이는지 확인). forward/loss/align 은 T 자동 처리.
            if args.mix_frames and global_rank == 0:
                logger.info(f"[mix] step {current_step}: frames={inputs.shape[2]} batch={inputs.shape[0]}")

            # [DEBUG] per-step memory logging
            if global_rank == 0:
                torch.cuda.reset_peak_memory_stats(rank)
                _mem_step_start = torch.cuda.memory_allocated(rank) / 1e9
                logger.info(f"[mem] step {current_step} start: allocated={_mem_step_start:.2f}GB")

            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False

            assert (
                step_gen or step_dis
            ), "You should backward either Gen or Dis in a step."

            with torch.cuda.amp.autocast(dtype=precision):
                # [MODIFIED - oliviaa/dit_align] GeopriorDiTAlignModel.forward() → 4-tuple
                recon, mu, log_var, z_cat = model(inputs)
                posterior = DiagonalGaussianDistribution(torch.cat([mu, log_var], dim=1))
                wavelet_coeffs = None

            # Generator Step
            if step_gen:
                with torch.cuda.amp.autocast(dtype=precision):
                    g_loss, g_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=0,
                        global_step=optimizer_step,
                        last_layer=gan_last_layer,
                        wavelet_coeffs=wavelet_coeffs,
                        split="train",
                    )

                # [NEW - mixed-length] per-duration rec_loss (dense, 매 gen step) — 81f/17f 따로 wandb.
                #   sparse 한 gnorm(10step+81f) 과 달리 81f loss 궤적을 dense 하게 추적.
                if args.mix_frames and global_rank == 0:
                    _mtag = "81f" if inputs.shape[2] >= args.mix_81_num_frames else "17f"
                    try:
                        wandb.log({f"mix/rec_loss_{_mtag}": float(g_log['train/rec_loss'])}, step=optimizer_step)
                    except Exception:
                        pass

                # ─── [NEW - oliviaa/dit_align] DiT dual-branch alignment ───
                align_loss = torch.tensor(0.0, device=rank)
                align_per_layer = {}
                _loss_diff = None  # [NEW] diffusion (flow matching) loss, no_fused path 에서만 set
                _diff_z_main = _diff_z_prior = 0.0
                if args.align_weight > 0:
                    _align_bs = args.align_batch_size if args.align_batch_size > 0 else inputs.shape[0]
                    _align_bs = min(_align_bs, inputs.shape[0])
                    inputs_align = inputs[:_align_bs]
                    z_cat_align = z_cat[:_align_bs]
                    # [align-equiv 검증] 고정 fixture 주입 (env ALIGN_EQUIV_FIXTURE) — stage2 와 동일 입력(z 동일).
                    import os as _os_eq
                    _equiv_fx = _os_eq.environ.get("ALIGN_EQUIV_FIXTURE")
                    _equiv_tidx = None
                    if _equiv_fx and _os_eq.path.exists(_equiv_fx):
                        _fx = torch.load(_equiv_fx, map_location="cpu")
                        inputs_align = _fx["inputs_align"].to(device=rank, dtype=precision)
                        z_cat_align = _fx["z_cat"].to(device=rank, dtype=precision).requires_grad_(True)
                        _equiv_tidx = int(_fx["tidx"])
                    with torch.cuda.amp.autocast(dtype=precision):
                        # Timestep 샘플링
                        if _equiv_tidx is not None:
                            tidx = _equiv_tidx
                        elif args.dit_timestep_mode == 'random':
                            tidx = random.randint(0, len(scheduler.timesteps) - 1)
                        else:
                            # fixed: 미리 정한 timestep 순환
                            _fixed_ts = [int(x) for x in args.dit_fixed_timesteps.split(",")]
                            tidx = _fixed_ts[current_step % len(_fixed_ts)]
                        timestep = scheduler.timesteps[tidx]
                        t_tensor = torch.tensor([timestep], device=rank, dtype=precision)
                        # [RESTORE matchS2] 위 기존 블록은 그대로 유지. flag off면 아래 if 안 타고 _align_sched=scheduler·
                        #   _sample_weight=None → downstream 완전 동일(byte-identical). on이면 stage2와 동일하게 override:
                        #   1000-step training sched + per-sample timestep + bsmntw weight (run_student_diffusion_forward 미러).
                        _align_sched = scheduler
                        _sample_weight = None
                        if getattr(args, 'align_match_stage2', False) and _equiv_tidx is None:
                            _align_sched = align_match_scheduler
                            _nst = len(_align_sched.timesteps)
                            _minb = int(args.align_min_timestep_boundary * _nst)
                            _maxb = max(_minb + 1, int(args.align_max_timestep_boundary * _nst))
                            _Bal = z_cat_align.shape[0]
                            tidx = torch.randint(_minb, _maxb, (_Bal,))
                            timestep = _align_sched.timesteps[tidx].to(device=rank, dtype=precision)   # (B,)
                            t_tensor = timestep                                                        # (B,)
                            _sample_weight = _align_sched.training_weight(timestep).to(device=rank, dtype=precision)  # (B,) bsmntw

                        # Decide the align_weight passed into the fused forward.
                        # - 2-backward mode: align_loss must be RAW (no weighting); the train
                        #   loop multiplies by w * align_weight after compute_adaptive_weight_2bwd.
                        # - single-backward adaptive: pass 1.0; _AdaptiveWeightingFn handles scaling.
                        # - no adaptive: pass args.align_weight directly.
                        if getattr(args, 'use_2backward_adaptive', False):
                            _aw_for_fused = 1.0
                        elif args.align_adaptive_weight:
                            _aw_for_fused = 1.0
                        else:
                            _aw_for_fused = args.align_weight

                        if getattr(args, 'no_fused_align', False):
                            # [NEW] Origin (kk4aiuyq) 식: run_teacher → run_student → compute_alignment_loss.
                            # fused 의 per-block detach + AlignGradInjector 대신 sum(grad-tracked losses).backward().
                            # [NEW] teacher_frozen_pretrained: teacher forward 동안만 dit 를 init pretrained Wan 으로 갈아끼움.
                            #   (a) LoRA adapter off, (b) block_norms/mod → pretrained snapshot. 직후 원복.
                            _teacher_bn_cur = {}
                            if _teacher_frozen_pretrained:
                                for _m in dit.modules():
                                    if isinstance(_m, BaseTunerLayer):
                                        _m.enable_adapters(False)              # LoRA off → base Wan body
                                for _p in _block_norms_mod_params:
                                    _teacher_bn_cur[id(_p)] = _p.data.clone()  # 학습된 현재값 백업
                                    _p.data.copy_(_teacher_bn0[id(_p)])        # pretrained 값으로 swap
                            if _equiv_fx:
                                torch.manual_seed(10101)  # teacher noise_ref 고정 (stage1/stage2 동일)
                            features_ref, grid_ref, context, t_mod, freqs_ref, _teacher_noise = run_teacher_forward(
                                dit=dit, dit_pipe=dit_pipe,
                                inputs_align=inputs_align,
                                scheduler=_align_sched, timestep=timestep, t_tensor=t_tensor,
                                null_context=null_context,
                                _align_block_set=_align_block_set,
                                _dit_offload=_dit_offload, _dit_fsdp2=_dit_fsdp2,
                                align_after_patchify=args.align_after_patchify,
                                rank=rank, precision=precision,
                                _use_t5_cache=_use_t5_cache, t5_cache=t5_cache,
                                _use_caption=_use_caption, caption_map=caption_map,
                                batch=batch, _align_bs=_align_bs,
                                logger=logger if global_rank == 0 else None,
                            )
                            # [NEW] teacher forward 끝 → dit 를 학습 상태로 원복 (student/diffusion forward 는 학습된 weight 사용).
                            if _teacher_frozen_pretrained:
                                for _p in _block_norms_mod_params:
                                    _p.data.copy_(_teacher_bn_cur[id(_p)])     # 학습된 block_norms 복원
                                for _m in dit.modules():
                                    if isinstance(_m, BaseTunerLayer):
                                        _m.enable_adapters(True)               # LoRA on → student denoiser 정상
                            # [REPA-E 정석] align forward 동안 DiT(denoiser) freeze → align grad 가 VAE encoder 로만 흐름.
                            # student_patchify + LoRA + block_norms_mod (= diffusion 과 공유되는 DiT param) 의 requires_grad 끔.
                            # weight 는 frozen 이지만 activation grad 는 통과 → z_cat 거쳐 VAE 로 align grad 전달. 충돌 차단.
                            _dit_align_frozen = []
                            if getattr(args, 'align_stop_grad_dit', False):
                                _cand = list(student_patchify.parameters()) if student_patchify is not None else []
                                _cand += list(dit.parameters())
                                for _p in _cand:
                                    if _p.requires_grad:
                                        _p.requires_grad_(False)
                                        _dit_align_frozen.append(_p)
                            if _equiv_fx:
                                torch.manual_seed(20202)  # student noise_cat 고정 (stage1/stage2 동일)
                            features_stu, grid_stu, noisy_cat, _patchify_output_ref = run_student_forward(
                                student_patchify=student_patchify, dit=dit,
                                inputs_align=inputs_align, z_cat_align=z_cat_align,
                                scheduler=_align_sched, timestep=timestep,
                                context=context, t_mod=t_mod,
                                model_module=model.module, mask_mode=args.mask_mode,
                                _align_block_set=_align_block_set,
                                _dit_offload=_dit_offload,
                                _use_gc=getattr(args, 'use_grad_checkpoint', False),
                                grad_checkpoint_num_blocks=getattr(args, 'grad_checkpoint_num_blocks', 0),
                                align_after_patchify=args.align_after_patchify,
                                rank=rank, precision=precision,
                                retain_grads=(not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0),
                                logger=logger if global_rank == 0 else None,
                                teacher_noise=_teacher_noise,
                                teacher_subsample_noise_mode=getattr(args, 'teacher_subsample_noise_mode', 'off'),
                            )
                            align_loss, align_per_layer = compute_alignment_loss(
                                features_stu, features_ref,
                                grid_stu=grid_stu, grid_ref=grid_ref,
                                loss_type=args.align_loss_type,
                                selected_layers=[int(x) for x in args.align_layers.split(",")] if args.align_layers != "all" else None,
                                agg=args.align_agg,
                                align_projections=(None if _equiv_fx else getattr(model.module, 'align_projections', None)),
                                sample_weight=_sample_weight,  # [RESTORE matchS2] None(off)=기존 동작, (B,)=per-sample bsmntw
                            )
                            # [align-equiv 검증] dump (env ALIGN_EQUIV_DUMP_OUT). stage2 와 비교용. native 경로(y_cond 재encode).
                            _equiv_out = _os_eq.environ.get("ALIGN_EQUIV_DUMP_OUT") if _equiv_fx else None
                            if _equiv_out:
                                import sys as _sys_eq
                                # cond 재계산 (run_student_forward 내부와 동일) — stage2 y_cond 와 비교용.
                                with torch.no_grad():
                                    _ia = inputs_align
                                    _img_in = torch.zeros_like(_ia); _img_in[:, :, 0:1] = _ia[:, :, 0:1]
                                    _enc = model.module.vae.encode(_img_in, scale=None)
                                    _mu = _enc[0][0] if isinstance(_enc[0], tuple) else _enc[0]
                                    _mu = model.module._norm_zmain(_mu)
                                    _zp = model.module._norm_zprior(model.module.vae._encode_prior(_img_in))
                                    _imgzc = torch.cat([_mu, _zp], dim=1)
                                for _p in (list(student_patchify.parameters()) if student_patchify is not None else []):
                                    _p.grad = None
                                align_loss.backward()
                                _pg = None
                                if student_patchify is not None:
                                    _w = getattr(student_patchify, 'weight', None)
                                    if _w is not None and _w.grad is not None:
                                        _pg = float(_w.grad.norm().item())
                                _dump = {
                                    "features_ref": [f.detach().float().cpu() for f in features_ref],
                                    "features_stu": [f.detach().float().cpu() for f in features_stu],
                                    "grid_ref": tuple(grid_ref), "grid_stu": tuple(grid_stu),
                                    "align_loss": float(align_loss.detach().item()),
                                    "per_layer": {k: float(v.detach().item()) for k, v in align_per_layer.items()},
                                    "tidx": int(tidx), "timestep": float(timestep),
                                    "patch_grad_norm": _pg,
                                    "noisy_cat": noisy_cat.detach().float().cpu(),
                                    "patchify_out": _patchify_output_ref.detach().float().cpu(),
                                    "patch_w_norm": (float(student_patchify.weight.detach().float().norm().item()) if (student_patchify is not None and getattr(student_patchify, 'weight', None) is not None) else None),
                                    "patch_in_ch": (int(student_patchify.in_channels) if student_patchify is not None else None),
                                    "patch_w_shape": (tuple(student_patchify.weight.shape) if (student_patchify is not None and getattr(student_patchify, 'weight', None) is not None) else None),
                                    "z_cat_norm": float(z_cat_align.detach().float().norm().item()),
                                    "image_z_cat": _imgzc.detach().float().cpu(),
                                    "image_z_cat_norm": float(_imgzc.float().norm().item()),
                                }
                                torch.save(_dump, _equiv_out)
                                if global_rank == 0:
                                    print(f"[align-equiv] stage1 dump 저장: {_equiv_out} (align_loss={_dump['align_loss']:.6f})", flush=True)
                                _sys_eq.exit(0)
                            # align_weight 적용 (compute_alignment_loss 는 weight 안 받음 — train loop 에서 scale)
                            align_loss = _aw_for_fused * align_loss

                            # [NEW - oliviaa] ablation: diffusion_only → align_loss ×0.
                            # align forward 는 그대로 돌아서 z_cat dual-output 그래프 유지 (static_graph 안전),
                            # 하지만 total 에 더해질 때 0 → VAE/공유param 으로 align grad 안 흐름. diffusion 만 학습.
                            if getattr(args, 'diffusion_only', False):
                                align_loss = align_loss * 0.0

                            # [REPA-E 정석] align forward 끝 → DiT(denoiser) requires_grad 복원.
                            # diffusion forward 는 정상 학습 (denoising grad → student_patchify/LoRA/block_norms/head).
                            for _p in _dit_align_frozen:
                                _p.requires_grad_(True)

                            # [NEW - oliviaa] diffusion (flow matching) loss — REPA-E 방식 (z.detach()).
                            # no_fused path 에서만 (context/t_mod 사용 가능). z_cat detach → VAE 무관.
                            if getattr(args, 'use_diffusion_loss', False) and getattr(model.module, 'davae_head', None) is not None:
                                _loss_diff, _diff_z_main, _diff_z_prior = run_student_diffusion_forward(
                                    student_patchify=student_patchify, dit=dit,
                                    davae_head=model.module.davae_head,
                                    inputs_align=inputs_align, z_cat_align=z_cat_align,
                                    scheduler=diffusion_scheduler, context=context,
                                    model_module=model.module, mask_mode=args.mask_mode,
                                    _align_block_set=_align_block_set,
                                    _dit_offload=_dit_offload,
                                    _use_gc=getattr(args, 'use_grad_checkpoint', False),
                                    grad_checkpoint_num_blocks=getattr(args, 'grad_checkpoint_num_blocks', 0),
                                    rank=rank, precision=precision,
                                    z_dim=args.z_dim, prior_z_dim=_known.prior_z_dim,
                                    min_timestep_boundary=getattr(args, 'diffusion_min_timestep_boundary', 0.0),
                                    max_timestep_boundary=getattr(args, 'diffusion_max_timestep_boundary', 1.0),
                                    logger=logger if (global_rank == 0 and current_step % args.log_steps == 0) else None,
                                )
                        else:
                            align_loss, align_per_layer, noisy_cat, _patchify_output_ref = fused_dit_align_forward(
                                dit=dit, student_patchify=student_patchify, dit_pipe=dit_pipe,
                                inputs_align=inputs_align, z_cat_align=z_cat_align,
                                scheduler=_align_sched, timestep=timestep, t_tensor=t_tensor,
                                null_context=null_context, model_module=model.module, mask_mode=args.mask_mode,
                                _align_block_set=_align_block_set,
                                _dit_offload=_dit_offload, _dit_fsdp2=_dit_fsdp2,
                                _use_gc=getattr(args, 'use_grad_checkpoint', False),
                                grad_checkpoint_num_blocks=getattr(args, 'grad_checkpoint_num_blocks', 0),
                                align_after_patchify=args.align_after_patchify,
                                rank=rank, precision=precision,
                                align_weight=_aw_for_fused,
                                loss_type=args.align_loss_type,
                                selected_layers=[int(x) for x in args.align_layers.split(",")] if args.align_layers != "all" else None,
                                agg=args.align_agg,
                                _use_t5_cache=_use_t5_cache, t5_cache=t5_cache,
                                _use_caption=_use_caption, caption_map=caption_map,
                                batch=batch, _align_bs=_align_bs,
                                retain_grads=(not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0),
                                logger=logger if global_rank == 0 else None,
                                align_projections=getattr(model.module, 'align_projections', None),
                                teacher_subsample_noise_mode=getattr(args, 'teacher_subsample_noise_mode', 'off'),
                            )

                # [NEW] 2-backward adaptive (legacy path) — mutually exclusive with _AdaptiveWeightingFn.
                # Computes w via two autograd.grad(retain_graph=True) calls on encoder.head[-1].weight,
                # then assembles total_loss = g_loss + align_weight * w * align_loss.
                # Single-backward path leaves total_loss = g_loss + align_loss (scaling already in graph).
                _w_adaptive_2bwd = None
                _w_raw_2bwd = None
                if (getattr(args, 'use_2backward_adaptive', False)
                        and args.align_weight > 0
                        and align_loss.item() > 0):
                    _align_last_layer = model.module.vae.encoder.head[-1].weight  # AW.weight (= single-branch call 시 plain conv backward)
                    _w_adaptive_2bwd, _w_raw_2bwd, _2bwd_rec_norm_main, _2bwd_align_norm_main = compute_adaptive_weight_2bwd(
                        g_loss, align_loss, _align_last_layer,
                        max_weight=getattr(args, 'adaptive_max_weight', 1e4),
                    )
                    total_loss = g_loss + args.align_weight * _w_adaptive_2bwd * align_loss
                else:
                    total_loss = g_loss + align_loss

                # [NEW - oliviaa] diffusion loss 결합 (= REPA-E denoising_loss).
                # z_cat detach 라 VAE grad 무관 → student_patchify + LoRA + davae_head + block_norms_mod 만 업데이트.
                if _loss_diff is not None:
                    total_loss = total_loss + getattr(args, 'diffusion_loss_weight', 1.0) * _loss_diff

                # [v26 disabled] 2bwd compare 비활성
                _w_2bwd_compare = None
                if False:
                    _align_last_layer = model.module.vae.encoder.head[-1].weight  # AW.weight (= single-branch call 시 plain conv backward)
                    _w_2bwd_raw = None
                    _2bwd_rec_norm = None
                    _2bwd_align_norm = None
                    # [DIAG v25] allow_unused + 직접 fetch — true disconnect vs cast issue 식별
                    if global_rank == 0:
                        try:
                            _test = torch.autograd.grad(align_loss, _align_last_layer, retain_graph=True, allow_unused=True)[0]
                            _test_norm = 'None' if _test is None else f'{_test.norm().item():.6e}'
                            logger.info(f"[DIAG] step {current_step} align→AW: grad={_test_norm}, is_leaf={_align_last_layer.is_leaf}, requires_grad={_align_last_layer.requires_grad}")
                        except Exception as _de:
                            import traceback
                            logger.warning(f"[DIAG] step {current_step} align→AW raised: {type(_de).__name__}: {_de}\n{traceback.format_exc()}")
                    try:
                        _w_2bwd_compare, _w_2bwd_raw, _2bwd_rec_norm, _2bwd_align_norm = compute_adaptive_weight_2bwd(
                            g_loss, align_loss, _align_last_layer,
                            max_weight=getattr(args, 'adaptive_max_weight', 1e4),
                        )
                        if global_rank == 0 and current_step < 5:
                            logger.info(f"[B-fix verify] 2bwd compare OK at step {current_step}: w={_w_2bwd_compare.item():.6f}")
                    except Exception as _e:
                        import traceback
                        if global_rank == 0:
                            logger.warning(f"[B-fix verify] 2bwd compare FAIL at step {current_step}: {type(_e).__name__}: {_e}\n{traceback.format_exc()}")

                # [NEW - oliviaa/dit_align] gradient accumulation 지원
                scaled_loss = scaler.scale(total_loss / _accum)
                # [NEW] DDP no_sync: 중간 accum step 에서는 gradient sync 생략 → 통신 overhead 절감
                # current_step 은 0 부터 시작, backward 후 증가 → (current_step + 1) 패턴으로 off-by-one 방지
                _is_accum_step = (current_step + 1) % _accum == 0
                # [DEBUG] memory before backward
                if global_rank == 0:
                    _mem_pre_bwd = torch.cuda.memory_allocated(rank) / 1e9
                    logger.info(f"[mem] step {current_step} pre-backward: allocated={_mem_pre_bwd:.2f}GB")

                # [NEW v5 - oliviaa/B-fix verify] use_b_adaptive 시 K step 마다 의 retain_graph (= OOM 방지)
                # 이전 (v2): 매 step 의 retain_graph → autograd.grad × 6 의 graph 의 누적 → 50GB/step ↑ → step 5 의 OOM (= v14 의 실제 cause)
                # 해결: 250 step 의 1번 만 retain_graph + verify (= ckpt save 의 간격 과 같음)
                # [v26 disabled] verify metric 제거 — 별도 run 으로 (B) vs 2backward 비교
                _verify_now = False
                _retain_for_verify = False
                if _accum > 1 and not _is_accum_step:
                    with model.no_sync():
                        scaled_loss.backward(retain_graph=_retain_for_verify)
                else:
                    scaled_loss.backward(retain_graph=_retain_for_verify)

                # [DEBUG] memory after backward
                if global_rank == 0:
                    _mem_post_bwd = torch.cuda.memory_allocated(rank) / 1e9
                    _mem_peak = torch.cuda.max_memory_allocated(rank) / 1e9
                    logger.info(f"[mem] step {current_step} post-backward: allocated={_mem_post_bwd:.2f}GB, peak={_mem_peak:.2f}GB")

                # grad norm log (rank0 only, log_steps 마다 = 매스텝). (구 v27 K=10 하드코딩 제거)
                if global_rank == 0 and (current_step % args.log_steps == 0):
                    try:
                        _wrap = model.module if hasattr(model, 'module') else model
                        _vae = _wrap.vae
                        def _gn(params):
                            t = 0.0
                            for p in params:
                                if p.grad is not None:
                                    t += p.grad.detach().float().norm().item() ** 2
                            return t ** 0.5
                        _enc_body = [p for n, p in _vae.encoder.named_parameters() if 'head.2' not in n]
                        _enc_head = [_vae.encoder.head[-1].weight, _vae.encoder.head[-1].bias]
                        _dec = list(_vae.decoder.parameters())
                        _lora = [p for n, p in dit.named_parameters() if 'lora_' in n.lower()]
                        _patch = list(student_patchify.parameters()) if student_patchify is not None else []
                        # [NEW] align_projections 학습 진행 추적 (= grad 흐르는지 + zero init 에서 weight 변화)
                        _ap_module = getattr(model.module, 'align_projections', None)
                        _ap = list(_ap_module.parameters()) if _ap_module is not None else []
                        _log = {
                            "gnorm/encoder_body":    _gn(_enc_body),
                            "gnorm/encoder_head":    _gn(_enc_head),
                            "gnorm/decoder":         _gn(_dec),
                            "gnorm/dit_lora":        _gn(_lora),
                            "gnorm/student_patchify": _gn(_patch),
                        }
                        if _ap:
                            _log["gnorm/align_projections"] = _gn(_ap)
                            # weight norm: zero init → 학습 진행 시 nonzero 증가
                            _ap_wn_sq = sum(p.detach().float().norm().item() ** 2
                                            for p in _ap if p.requires_grad)
                            _log["weight/align_projections_norm"] = _ap_wn_sq ** 0.5
                        # [NEW - jeeyoung] encoder.head 에서 rec/align gradient norm 분리 로깅.
                        #   gnorm/encoder_head = 둘 합산값. 분리값(합산 전)은 AdaptiveWeightedConv3dFn 가 저장.
                        try:
                            from adaptive_weighted_causal_conv_3d import _AdaptiveWeightedConv3dFn as _AWF
                            if getattr(_AWF, '_last_grad_W_main_norm', None) is not None:
                                _log["gnorm/encoder_head_rec"]   = float(_AWF._last_grad_W_main_norm.item())
                                _log["gnorm/encoder_head_align"] = float(_AWF._last_grad_W_adv_norm.item())
                        except Exception:
                            pass
                        # [NEW - mixed-length B] 81f가 decoder+align 둘 다 학습시키는지 검증 (프레임수별 분리).
                        #   mix/align_gnorm_81f > 0 = 81f batch가 align projection까지 grad 흘림 (B 핵심).
                        if args.mix_frames:
                            _tag = "81f" if inputs.shape[2] >= args.mix_81_num_frames else "17f"
                            _log[f"mix/decoder_gnorm_{_tag}"] = _log["gnorm/decoder"]
                            if _ap:
                                _log[f"mix/align_gnorm_{_tag}"] = _log["gnorm/align_projections"]
                        # [NEW] diffusion: davae_head grad norm (= diffusion grad 흐름 검증)
                        _dh_module = getattr(model.module, 'davae_head', None)
                        if _dh_module is not None:
                            _dh = [p for p in _dh_module.parameters() if p.requires_grad]
                            if _dh:
                                _log["gnorm/davae_head"] = _gn(_dh)
                                # [NEW] davae_head weight norm: grad 흐름 → optimizer step → weight 변하는지 검증.
                                # head_main 은 zero init → 학습되면 head_main norm 증가해야 함.
                                _dh_named = {n: p for n, p in _dh_module.named_parameters() if p.requires_grad}
                                _hm = [p for n, p in _dh_named.items() if 'head_main' in n]
                                _hp = [p for n, p in _dh_named.items() if 'head_prior' in n]
                                if _hm:
                                    _log["weight/davae_head_main_norm"] = (sum(p.detach().float().norm().item()**2 for p in _hm))**0.5
                                if _hp:
                                    _log["weight/davae_head_prior_norm"] = (sum(p.detach().float().norm().item()**2 for p in _hp))**0.5
                        wandb.log(_log, step=optimizer_step)
                    except Exception as _ge:
                        if current_step < 5:
                            logger.warning(f"[gnorm] fail: {_ge}")

                # [NEW v5 - oliviaa/B-fix verify] K step 마다 의 verify (= OOM 방지)
                # 매 step 의 verify (= autograd.grad × 6 + retain_graph) → 메모리 누적 → step 5 의 OOM (= v14)
                # 해결: _verify_now (= 250 step 의 1번) 일 때 만 verify block
                if _verify_now:
                    try:
                        _wrapper = model.module if hasattr(model, 'module') else model
                        _vae = _wrapper.vae
                        _opt_step = optimizer_step

                        def _gnorm(params):
                            total = 0.0
                            cnt = 0
                            for p in params:
                                if p.grad is not None:
                                    total += p.grad.detach().float().norm().item() ** 2
                                    cnt += 1
                            return (total ** 0.5) if cnt > 0 else 0.0

                        _enc_body = [p for n, p in _vae.encoder.named_parameters() if 'head.2' not in n]
                        _enc_head = [_vae.encoder.head[-1].weight, _vae.encoder.head[-1].bias]
                        _dec = list(_vae.decoder.parameters())
                        _lora = [p for n, p in dit.named_parameters() if 'lora_' in n.lower()]
                        _patch = list(student_patchify.parameters()) if student_patchify is not None else []

                        # 1. grad norm log = rank0 only (= 단 local norm, all_reduce X)
                        if global_rank == 0:
                            wandb.log({
                                "verify/grad_norm/encoder_body":    _gnorm(_enc_body),
                                "verify/grad_norm/encoder_head_last": _gnorm(_enc_head),
                                "verify/grad_norm/decoder":         _gnorm(_dec),
                                "verify/grad_norm/dit_lora":        _gnorm(_lora),
                                "verify/grad_norm/student_patchify": _gnorm(_patch),
                            }, step=_opt_step)

                        # 2. (B) vs 2backward 의 직접 grad diff
                        # 모든 rank 가 autograd.grad 호출 (= 우리 _AdaptiveWeightedConv3dFn.backward 의 dist.all_reduce 의 모든 rank 의 호출 의 보장)
                        # log / 비교 는 rank0 만
                        if _w_2bwd_compare is not None:
                            _c = float(_w_2bwd_compare.item())

                            def _grad_diff(params_to_compare, label):
                                _trainable = [p for p in params_to_compare if p.requires_grad]
                                if not _trainable:
                                    return
                                # 모든 rank 가 호출 (= DDP collective sync)
                                _rec_g = torch.autograd.grad(g_loss, _trainable, retain_graph=True, allow_unused=True)
                                _align_g = torch.autograd.grad(align_loss, _trainable, retain_graph=True, allow_unused=True)
                                if global_rank != 0:
                                    return
                                _diff_sq, _norm_sq, _max_rel = 0.0, 0.0, 0.0
                                for _p, _gr, _ga in zip(_trainable, _rec_g, _align_g):
                                    if _p.grad is None or _gr is None or _ga is None:
                                        continue
                                    _g_2bwd = _gr + _c * _ga
                                    _g_b = _p.grad
                                    _d = (_g_b - _g_2bwd).norm().item()
                                    _n = _g_2bwd.norm().item()
                                    _diff_sq += _d ** 2
                                    _norm_sq += _n ** 2
                                    if _n > 1e-10:
                                        _max_rel = max(_max_rel, _d / _n)
                                _rel = (_diff_sq ** 0.5) / max(_norm_sq ** 0.5, 1e-10)
                                wandb.log({
                                    f"verify/{label}_rel_diff": _rel,
                                    f"verify/{label}_max_rel":  _max_rel,
                                }, step=_opt_step)

                            # 1. encoder.head[-1] = 우리 manual 의 직접 grad (★ 가장 critical)
                            _grad_diff([_vae.encoder.head[-1].weight, _vae.encoder.head[-1].bias], "encoder_head_last")
                            # 2. encoder body = 우리 grad_x 의 chain
                            _grad_diff(_enc_body, "encoder_body")
                            # 3. decoder = sanity check (= 우리 manual 와 무관)
                            _grad_diff(_dec, "decoder")
                    except Exception as _e:
                        if current_step < 5 and global_rank == 0:
                            logger.warning(f"[B-fix verify] fail: {_e}")

                # accumulation 완료 시에만 optimizer step
                if _is_accum_step:
                    # [FIX - FSDP2 호환] DDP 밖 param 들의 grad 수동 sync.
                    #   단 grad 가 DTensor 면 = FSDP2 가 reduce-scatter 로 이미 동기화함 → 수동 all_reduce 스킵
                    #   (DTensor 에 dist.all_reduce 하면 NotImplementedError). 일반 텐서(FSDP2 미wrap)만 수동 sync.
                    #   dit_fsdp2 OFF 시: 전부 일반 텐서 → 모두 all_reduce = 기존 동작과 byte-identical.
                    try:
                        from torch.distributed.tensor import DTensor as _DTensor
                    except Exception:
                        _DTensor = ()
                    def _manual_sync(g):
                        return g is not None and not isinstance(g, _DTensor)
                    # [FIX - oliviaa] student_patchify 는 DDP 밖이라 gradient 수동 sync
                    if dist.get_world_size() > 1 and student_patchify is not None:
                        for p in student_patchify.parameters():
                            if _manual_sync(p.grad):
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    # [FIX - oliviaa] dit LoRA 도 DDP 밖 (dit 는 DDP wrap 안 됨) → gradient 수동 sync.
                    # 이전엔 LoRA all_reduce 없어 8 GPU 가 각자 학습 (= effective batch = per-GPU batch,
                    # ckpt 는 rank0 LoRA 만 저장 → 나머지 학습 버려짐). 이제 8 GPU 평균 (= effective batch 정상).
                    if dist.get_world_size() > 1 and getattr(args, 'use_lora', False) and dit is not None:
                        for n, p in dit.named_parameters():
                            if 'lora_' in n and _manual_sync(p.grad):
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    # [NEW - oliviaa] diffusion: block_norms_mod 도 dit param (DDP 밖) → 수동 sync
                    if dist.get_world_size() > 1 and _block_norms_mod_params:
                        for p in _block_norms_mod_params:
                            if _manual_sync(p.grad):
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                    # [NEW - oliviaa] gradient clipping + norm 로깅
                    scaler.unscale_(gen_optimizer)
                    # [NEW - jeeyoung] per-component grad clip (decoder/encoder_head/encoder_body 따로 조절, arg)
                    #   각 그룹 따로 clip + pre/post norm 로깅(clip 잘 되는지 검증, return값이라 추가비용 ~0).
                    _vae_c = model.module.vae if hasattr(model.module, 'vae') else model.module
                    _do_clip_log = (not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0)
                    _clip_log = {}
                    for _cname, _cparams, _cval in [
                        ('decoder',      list(_vae_c.decoder.parameters()),                                        getattr(args, 'decoder_grad_clip', 0.0)),
                        ('encoder_head', [_vae_c.encoder.head[-1].weight, _vae_c.encoder.head[-1].bias],           getattr(args, 'encoder_head_grad_clip', 0.0)),
                        ('encoder_body', [p for n, p in _vae_c.encoder.named_parameters() if 'head.2' not in n],  getattr(args, 'encoder_body_grad_clip', 0.0)),
                        ('align_projections', list(model.module.align_projections.parameters()) if getattr(model.module, 'align_projections', None) is not None else [], getattr(args, 'align_projections_grad_clip', 0.0)),
                        ('student_patchify',  list(student_patchify.parameters()) if student_patchify is not None else [],                                              getattr(args, 'student_patchify_grad_clip', 0.0)),
                    ]:
                        if _cval > 0 and _cparams:
                            _pre = torch.nn.utils.clip_grad_norm_(_cparams, _cval)   # clip 실행 + pre-clip norm 반환
                            if _do_clip_log:
                                _clip_log[f"clip/{_cname}_preclip"] = float(_pre.item())
                    if _clip_log:
                        try:
                            wandb.log(_clip_log, step=optimizer_step)
                        except Exception as _ce:
                            logger.warning(f"[clip] wandb log 실패: {_ce}")
                    elif _do_clip_log:
                        logger.warning(f"[clip] _clip_log 비어있음 (decoder_clip={getattr(args,'decoder_grad_clip',0)})")
                    # 기존 whole-model max_grad_norm (default 0=off, 호환 유지)
                    _max_grad_norm = getattr(args, 'max_grad_norm', 0.0)
                    if _max_grad_norm > 0:
                        _all_params = list(model.parameters())
                        if student_patchify is not None:
                            _all_params += list(student_patchify.parameters())
                        torch.nn.utils.clip_grad_norm_(_all_params, _max_grad_norm)
                    if not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0:
                        def _grad_norm(params):
                            grads = [p.grad for p in params if p.grad is not None]
                            if not grads:
                                return 0.0
                            return torch.cat([g.flatten() for g in grads]).norm().item()

                        _grad_log = {
                            "grad/encoder": _grad_norm(vae_module.encoder.parameters()),
                            "grad/decoder": _grad_norm(vae_module.decoder.parameters()),
                            "grad/conv1": _grad_norm(vae_module.conv1.parameters()),
                        }
                        if getattr(args, 'use_lora', False) and dit is not None:
                            _grad_log["grad/lora"] = _grad_norm(
                                p for n, p in dit.named_parameters() if 'lora_' in n
                            )
                        # [FIX] z_dim/prior_z_dim 동적 — 이전 32/48 hardcoded 였음.
                        _zd = args.z_dim
                        _pzd = _known.prior_z_dim
                        _np = _zd + _pzd  # noisy_z_prior end (= z_dim + prior_z_dim)
                        if student_patchify is not None and student_patchify.weight.grad is not None:
                            _w = student_patchify.weight.grad
                            # pretrained 부분 = noisy_z_prior + image_z_prior (둘 다 prior 채널)
                            _pre = torch.cat([_w[:, _zd:_np], _w[:, -_pzd:]], dim=1)
                            # added 부분 = noisy_z_main + mask + image_z_main
                            _add = torch.cat([_w[:, :_zd], _w[:, _np:-_pzd]], dim=1)
                            _grad_log["grad/patchify_pretrained"] = _pre.norm().item()
                            _grad_log["grad/patchify_added"] = _add.norm().item()
                            _grad_log["grad/patchify_ratio"] = (_add.norm() / (_pre.norm() + 1e-10)).item()
                        if 'noisy_cat' in dir() and hasattr(noisy_cat, 'grad') and noisy_cat.grad is not None:
                            _grad_log["grad/input_z_main"] = noisy_cat.grad[:, :_zd].norm().item()
                            _grad_log["grad/input_z_prior"] = noisy_cat.grad[:, _zd:_np].norm().item()
                            _grad_log["grad/input_ratio"] = (noisy_cat.grad[:, :_zd].norm() / (noisy_cat.grad[:, _zd:_np].norm() + 1e-10)).item()
                        if 'z_cat_align' in dir() and hasattr(z_cat_align, 'grad') and z_cat_align.grad is not None:
                            _grad_log["grad/zcat_z_main"] = z_cat_align.grad[:, :_zd].norm().item()
                            _grad_log["grad/zcat_z_prior"] = z_cat_align.grad[:, _zd:_np].norm().item()
                        elif hasattr(z_cat, 'grad') and z_cat.grad is not None:
                            _grad_log["grad/zcat_z_main"] = z_cat.grad[:, :_zd].norm().item()
                            _grad_log["grad/zcat_z_prior"] = z_cat.grad[:, _zd:_np].norm().item()
                        if '_patchify_output_ref' in dir() and hasattr(_patchify_output_ref, 'grad') and _patchify_output_ref.grad is not None:
                            _grad_log["grad/patchify_output"] = _patchify_output_ref.grad.norm().item()
                        wandb.log(_grad_log, step=optimizer_step)

                    # [NEW - warmup] linear LR warmup: 이 run 시작(_warmup_anchor_step) 기준 N optimizer step 동안 0->target ramp.
                    #   각 param_group 의 목표 lr(_warmup_target_lrs)에 factor 곱함. warmup 끝나면 factor=1 (target 고정).
                    if args.warmup_steps > 0:
                        _wf = min(1.0, max(0, optimizer_step - _warmup_anchor_step) / float(args.warmup_steps))
                        for _i, _pg in enumerate(gen_optimizer.param_groups):
                            _pg['lr'] = _warmup_target_lrs[_i] * _wf
                        if global_rank == 0 and current_step % args.log_steps == 0:
                            try:
                                wandb.log({"lr/vae": gen_optimizer.param_groups[0]['lr'], "lr/warmup_factor": _wf}, step=optimizer_step)
                            except Exception:
                                pass

                    scaler.step(gen_optimizer)
                    scaler.update()
                    gen_optimizer.zero_grad()
                    if args.ema:
                        ema.update()
                # [NEW] grad accumulation: loss 누적 (accum>1 일 때)
                if _accum > 1:
                    _loss_accum["g_loss"] += g_loss.item() / _accum
                    _loss_accum["rec_loss"] += g_log['train/rec_loss'] / _accum
                    _loss_accum["kl_loss"] += g_log['train/kl_loss'] / _accum
                    _loss_accum["nll_loss"] += g_log['train/nll_loss'] / _accum
                    _loss_accum["align_loss"] += align_loss.item() / _accum
                    _loss_accum["total_loss"] += total_loss.item() / _accum

                # accum=1 이면 매 step 로깅, accum>1 이면 optimizer step 시에만 로깅
                _should_log = global_rank == 0 and current_step % args.log_steps == 0
                if _accum > 1:
                    _should_log = _should_log and _is_accum_step

                if _should_log:
                    if _accum > 1:
                        _gl = _loss_accum["g_loss"]
                        _rl = _loss_accum["rec_loss"]
                        _kl = _loss_accum["kl_loss"]
                        _nl = _loss_accum["nll_loss"]
                        _al = _loss_accum["align_loss"]
                        _tl = _loss_accum["total_loss"]
                    else:
                        _gl = g_loss.item()
                        _rl = g_log['train/rec_loss']
                        _kl = g_log['train/kl_loss']
                        _nl = g_log['train/nll_loss']
                        _al = align_loss.item()
                        _tl = total_loss.item()

                    latents_std = posterior.sample().std().item()
                    last_metrics["gen_loss"] = f"{_gl:.4f}"
                    last_metrics["rec_loss"] = f"{_rl:.4f}"
                    last_metrics["nll_loss"] = f"{_nl:.4f}"
                    last_metrics["kl_loss"] = f"{_kl:.6f}"
                    last_metrics["latents_std"] = f"{latents_std:.4f}"
                    wandb.log({"train/generator_loss": _gl}, step=optimizer_step)
                    wandb.log({"train/rec_loss": _rl}, step=optimizer_step)
                    wandb.log({"train/kl_loss": _kl}, step=optimizer_step)
                    wandb.log({"train/nll_loss": _nl}, step=optimizer_step)
                    wandb.log({"train/latents_std": latents_std}, step=optimizer_step)
                    wandb.log({"train/g_loss": g_log.get('train/g_loss', 0)}, step=optimizer_step)
                    wandb.log({"train/d_weight": g_log.get('train/d_weight', 0)}, step=optimizer_step)
                    if 'train/sb_loss' in g_log:
                        wandb.log({"train/sb_loss": g_log['train/sb_loss']}, step=optimizer_step)
                    if 'train/wl_loss' in g_log:
                        wandb.log({"train/wl_loss": g_log['train/wl_loss']}, step=optimizer_step)
                    wandb.log({"train/align_loss": _al}, step=optimizer_step)
                    wandb.log({"train/total_loss": _tl}, step=optimizer_step)
                    # [NEW - oliviaa] diffusion loss 로깅 (= REPA-E denoising_loss + z_main/z_prior 분리)
                    if _loss_diff is not None:
                        wandb.log({
                            "train/diffusion_loss": float(_loss_diff.item()),
                            "train/diff_loss_z_main": _diff_z_main,
                            "train/diff_loss_z_prior": _diff_z_prior,
                        }, step=optimizer_step)
                    # [REVIVED] align_loss_weighted — actual magnitude flowing through backward.
                    # adaptive paths: w sourced from active mode (2-backward vs single-backward).
                    if getattr(args, 'log_adaptive_weight', False):
                        if getattr(args, 'use_2backward_adaptive', False) and _w_adaptive_2bwd is not None:
                            _w_log = float(_w_adaptive_2bwd.item() if hasattr(_w_adaptive_2bwd, 'item') else _w_adaptive_2bwd)
                            if _w_raw_2bwd is not None:
                                wandb.log({"train/adaptive_weight_w_raw": float(_w_raw_2bwd.item())}, step=optimizer_step)
                        elif getattr(args, 'use_b_adaptive', False):
                            # [NEW - oliviaa/B-fix verify] (B) 의 ratio = _AdaptiveWeightedConv3dFn._last_c
                            try:
                                from adaptive_weighted_causal_conv_3d import _AdaptiveWeightedConv3dFn
                                _w_log = float(_AdaptiveWeightedConv3dFn._last_c.item())
                            except (ImportError, AttributeError):
                                _w_log = 1.0
                        elif args.align_adaptive_weight and getattr(_AdaptiveWeightingFn, '_last_c', None) is not None:
                            _w_log = float(_AdaptiveWeightingFn._last_c.item())
                        else:
                            _w_log = 1.0
                        wandb.log({"train/adaptive_weight_w": _w_log}, step=optimizer_step)
                        wandb.log({"train/align_loss_weighted": args.align_weight * _w_log * _al}, step=optimizer_step)
                        # [NEW] adaptive_weight_w_raw (= clamp 전) + grad_W norm 매 step log
                        # ratio = ||grad_W_main|| / (||grad_W_adv|| + eps). 분모(adv) 급락 시 ratio 폭증 → clamp → spike.
                        # z_prior noise spike 원인 추적: adv norm 이 spike 직전 vanish 하는지 확인.
                        try:
                            from adaptive_weighted_causal_conv_3d import _AdaptiveWeightedConv3dFn as _Fn
                            if getattr(_Fn, '_last_c_raw', None) is not None:
                                wandb.log({"train/adaptive_weight_w_raw": float(_Fn._last_c_raw.item())}, step=optimizer_step)
                            if getattr(_Fn, '_last_grad_W_main_norm', None) is not None:
                                wandb.log({
                                    "debug/grad_W_main_norm": float(_Fn._last_grad_W_main_norm.item()),
                                    "debug/grad_W_adv_norm":  float(_Fn._last_grad_W_adv_norm.item()),
                                    "debug/grad_y_main_norm": float(_Fn._last_grad_y_main_norm.item()),
                                    "debug/grad_y_adv_norm":  float(_Fn._last_grad_y_adv_norm.item()),
                                }, step=optimizer_step)
                        except (ImportError, AttributeError):
                            pass

                        # [v27c] 두 run 비교용 — 분자/분모 + W norm log
                        try:
                            _wrap2 = model.module if hasattr(model, 'module') else model
                            _hwl = _wrap2.vae.encoder.head[-1].weight
                            wandb.log({"weight/head_last_norm": _hwl.detach().float().norm().item()}, step=optimizer_step)
                        except Exception:
                            pass
                        if getattr(args, 'use_b_adaptive', False):
                            try:
                                from adaptive_weighted_causal_conv_3d import _AdaptiveWeightedConv3dFn as _Fn
                                if getattr(_Fn, '_last_grad_W_main_norm', None) is not None:
                                    wandb.log({
                                        "ratio/rec_grad_W_norm":   float(_Fn._last_grad_W_main_norm.item()),
                                        "ratio/align_grad_W_norm": float(_Fn._last_grad_W_adv_norm.item()),
                                    }, step=optimizer_step)
                            except Exception:
                                pass
                        elif getattr(args, 'use_2backward_adaptive', False) and '_2bwd_rec_norm_main' in dir() and _2bwd_rec_norm_main is not None:
                            wandb.log({
                                "ratio/rec_grad_W_norm":   float(_2bwd_rec_norm_main.item()),
                                "ratio/align_grad_W_norm": float(_2bwd_align_norm_main.item()),
                            }, step=optimizer_step)

                        # [NEW - oliviaa/B-fix verify] (B) 학습 의 case — 2bwd 의 ratio 도 log + diff 비교
                        if getattr(args, 'use_b_adaptive', False) and _w_2bwd_compare is not None:
                            _w_2bwd_val = float(_w_2bwd_compare.item() if hasattr(_w_2bwd_compare, 'item') else _w_2bwd_compare)
                            _rel_diff = abs(_w_log - _w_2bwd_val) / max(_w_2bwd_val, 1e-10)
                            _log_d = {
                                "train/adaptive_weight_2bwd_compare": _w_2bwd_val,
                                "train/adaptive_weight_b_vs_2bwd_diff_abs": abs(_w_log - _w_2bwd_val),
                                "train/adaptive_weight_b_vs_2bwd_diff_rel": _rel_diff,
                            }
                            # raw ratio (= clamp 전) log — saturation 영향 제외 한 진짜 비교
                            if _w_2bwd_raw is not None:
                                _w_2bwd_raw_val = float(_w_2bwd_raw.item())
                                _log_d["train/adaptive_weight_2bwd_raw"] = _w_2bwd_raw_val
                            from adaptive_weighted_causal_conv_3d import _AdaptiveWeightedConv3dFn as _Fn
                            if getattr(_Fn, '_last_c_raw', None) is not None:
                                _w_b_raw_val = float(_Fn._last_c_raw.item())
                                _log_d["train/adaptive_weight_b_raw"] = _w_b_raw_val
                                if _w_2bwd_raw is not None:
                                    _log_d["train/adaptive_weight_raw_diff_rel"] = abs(_w_b_raw_val - _w_2bwd_raw_val) / max(_w_2bwd_raw_val, 1e-10)
                            # [DEBUG v23] (B) AW backward 의 intermediate norm + 2backward norm 비교
                            if getattr(_Fn, '_last_grad_W_main_norm', None) is not None:
                                _log_d["debug/b_grad_W_main_norm"] = float(_Fn._last_grad_W_main_norm.item())
                                _log_d["debug/b_grad_W_adv_norm"] = float(_Fn._last_grad_W_adv_norm.item())
                                _log_d["debug/b_grad_y_main_norm"] = float(_Fn._last_grad_y_main_norm.item())
                                _log_d["debug/b_grad_y_adv_norm"] = float(_Fn._last_grad_y_adv_norm.item())
                            if _2bwd_rec_norm is not None:
                                _log_d["debug/2bwd_rec_norm"] = float(_2bwd_rec_norm.item())
                                _log_d["debug/2bwd_align_norm"] = float(_2bwd_align_norm.item())
                            wandb.log(_log_d, step=optimizer_step)

                    # [REMOVED v2 - oliviaa/B-fix verify] grad norm log 의 위치 이동 → backward 직후 (= line ~1547)
                    for layer_name, layer_loss in align_per_layer.items():
                        wandb.log({f"train/align_layer/{layer_name}": layer_loss.item()}, step=optimizer_step)

                    # [NEW] z_main BN running stats 로깅 (normalize_zmain_bn 활성 시)
                    if getattr(args, 'normalize_zmain_bn', False):
                        # GeopriorDiTAlignModel 는 DDP 안: model.module.zmain_bn (no DDP 면 model.zmain_bn)
                        _wrapper = model.module if hasattr(model, 'module') else model
                        _bn = _wrapper.zmain_bn if hasattr(_wrapper, 'zmain_bn') else None
                        if _bn is not None:
                            _rm = _bn.running_mean.detach()
                            _rv = _bn.running_var.detach()
                            # summary: per-channel mean/var 의 통계
                            wandb.log({
                                "bn_zmain/rm_mean": _rm.mean().item(),
                                "bn_zmain/rm_std": _rm.std().item(),
                                "bn_zmain/rm_absmax": _rm.abs().max().item(),
                                "bn_zmain/rv_mean": _rv.mean().item(),
                                "bn_zmain/rv_std": _rv.std().item(),
                                "bn_zmain/rv_min": _rv.min().item(),
                                "bn_zmain/rv_max": _rv.max().item(),
                            }, step=optimizer_step)
                            # per-channel detail
                            for c in range(_rm.shape[0]):
                                wandb.log({
                                    f"bn_zmain/rm_c{c:02d}": _rm[c].item(),
                                    f"bn_zmain/rv_c{c:02d}": _rv[c].item(),
                                }, step=optimizer_step)

                    # 누적 초기화
                    if _accum > 1:
                        for k in _loss_accum:
                            _loss_accum[k] = 0.0

            # Discriminator Step
            if step_dis:
                with torch.cuda.amp.autocast(dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=optimizer_step,
                        last_layer=None,
                        split="train",
                    )
                # [NEW] disc 에도 accumulation 적용 (gen 과 동일 빈도로 update)
                disc_scaler.scale(d_loss / _accum).backward()
                if _is_accum_step:
                    disc_scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(disc.module.discriminator.parameters(), 1.0)
                    disc_scaler.step(disc_optimizer)
                    disc_scaler.update()
                    disc_optimizer.zero_grad()
                    if global_rank == 0 and current_step % args.log_steps == 0:
                        last_metrics["disc_loss"] = f"{d_loss.item():.4f}"
                        wandb.log({"train/discriminator_loss": d_loss.item()}, step=optimizer_step)

            update_bar(bar)
            _was_accum_step = (current_step + 1) % _accum == 0  # backward 후 increment 전 의 _is_accum_step 과 동일
            current_step += 1
            # [FIX] optimizer_step 의 의 의 의 의 의 의 직접 increment (= ckpt 의 optim_step 복원 후 누적). accum 변경 시 일관성.
            if _was_accum_step:
                optimizer_step += 1

            def valid_model(model, name="", dataloader=None):
                set_eval(modules_to_train)
                _loader = dataloader if dataloader is not None else val_dataloader
                # [NEW - oliviaa/dit_align] dit_pipe 를 valid() 에 주입
                valid._dit_pipe = dit_pipe
                psnr_list, lpips_list, video_log, z_main_vecs, z_prior_vecs, z_cat_vecs, z_ref_vecs, pr, low_freq, noise_robust = valid(
                    global_rank, rank, model, _loader, precision, args,
                    lpips_model=shared_lpips_model,
                )
                valid_psnr, valid_lpips, valid_video_log, drift_metrics = gather_valid_result(
                    psnr_list, lpips_list, video_log, rank, dist.get_world_size(),
                    z_main_vecs=z_main_vecs, z_prior_vecs=z_prior_vecs,
                )
                # [NEW - oliviaa/dit_align] z_cat vs z_ref alignment CKA
                align_metrics = None
                if z_cat_vecs and z_ref_vecs:
                    gathered_z_cat = [None for _ in range(dist.get_world_size())]
                    gathered_z_ref = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(gathered_z_cat, torch.cat(z_cat_vecs, dim=0))
                    dist.all_gather_object(gathered_z_ref, torch.cat(z_ref_vecs, dim=0))
                    if rank == 0:
                        all_z_cat = torch.cat(gathered_z_cat, dim=0)
                        all_z_ref = torch.cat(gathered_z_ref, dim=0)
                        align_metrics = {
                            "cknna": compute_cknna(all_z_cat, all_z_ref, topk=10),
                            "linear_cka": compute_linear_cka(all_z_cat, all_z_ref),
                        }

                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    # video: wandb.Video accepts (N, T, C, H, W) or (T, C, H, W).
                    # tensor_to_video already returns (T, C, H, W); stacked → (N, T, C, H, W). No transpose needed.
                    _vid_arr = np.array(valid_video_log)
                    wandb.log({f"val{name}/recon": wandb.Video(_vid_arr, fps=10)}, step=optimizer_step)
                    wandb.log({f"val{name}/psnr": valid_psnr}, step=optimizer_step)
                    logger.info(f"[val-psnr] val{name}/psnr = {valid_psnr:.2f}dB  lpips = {valid_lpips:.4f}")  # [NEW] stdout 출력 (모니터링용)
                    wandb.log({f"val{name}/lpips": valid_lpips}, step=optimizer_step)
                    # [NEW - jeeyoung] decode frame 수 검증 — 학습 중 chunked-decode frame 손실(17->15 / 81->71) 없는지 모니터.
                    #   in-training 은 force_single_pass 라 손실 0 이어야 정상 (input_T == decode_T).
                    _fc = getattr(valid, '_frame_check', None)
                    if _fc is not None:
                        wandb.log({f"val{name}/input_frames": _fc[0], f"val{name}/decode_frames": _fc[1]}, step=optimizer_step)
                        logger.info(f"[frame-check] val{name}: input_T={_fc[0]} decode_T={_fc[1]} {'OK' if _fc[0]==_fc[1] else 'MISMATCH!!'}")
                    # [NEW - jeeyoung] SSVAE diffusability — 480x832x81(HD) eval 에서만 로깅 (256 base 는 skip).
                    #   "480" in name 으로 HD 판정 (name="_480x832x81" 또는 "_ema_480x832x81"). 256(EMA 포함) 제외.
                    #   diffusability 는 stage2 해상도(480x832x81)에서만 의미 (256 low_freq 는 corner 비율 달라 비교 불가).
                    if "480" in name:
                        wandb.log({f"val{name}/diffusability_pr": pr}, step=optimizer_step)
                        wandb.log({f"val{name}/diffusability_lowfreq": low_freq}, step=optimizer_step)
                    # [NEW] noise robustness (sigma 별, rank-averaged)
                    for _s, _v in (noise_robust or {}).items():
                        wandb.log({f"val{name}/noise_robust_s{_s}": _v}, step=optimizer_step)
                    # z_main↔z_prior drift
                    if drift_metrics is not None:
                        wandb.log({f"val{name}/cknna_z_drift":      drift_metrics["cknna"]},      step=optimizer_step)
                        wandb.log({f"val{name}/linear_cka_z_drift": drift_metrics["linear_cka"]}, step=optimizer_step)
                        if drift_metrics["cosine_sim"] is not None:
                            wandb.log({f"val{name}/cosine_sim_z_drift": drift_metrics["cosine_sim"]}, step=optimizer_step)
                        cos_str = f"{drift_metrics['cosine_sim']:.4f}" if drift_metrics['cosine_sim'] is not None else "N/A (dim mismatch)"
                        logger.info(
                            f"val{name} drift — CKNNA: {drift_metrics['cknna']:.4f} | "
                            f"CKA: {drift_metrics['linear_cka']:.4f} | "
                            f"cos: {cos_str}"
                        )
                    # [NEW - oliviaa/dit_align] z_cat↔z_ref alignment metrics
                    if align_metrics is not None:
                        wandb.log({f"val{name}/cknna_z_align":      align_metrics["cknna"]},      step=optimizer_step)
                        wandb.log({f"val{name}/linear_cka_z_align": align_metrics["linear_cka"]}, step=optimizer_step)
                        logger.info(
                            f"val{name} align — CKNNA: {align_metrics['cknna']:.4f} | "
                            f"CKA: {align_metrics['linear_cka']:.4f}"
                        )
                    logger.info(f"{name} Validation done.")

            if _is_accum_step and args.eval_video_path is not None and (optimizer_step % args.eval_steps == 0 or optimizer_step == 1 or not _initial_eval_done):
                if global_rank == 0:
                    logger.info("Starting validation...")
                valid_model(model)
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, "ema")
                    ema.restore()
                # [NEW - oliviaa] HD resolution evals
                for _res_tag, _hd_loader in hd_val_dataloaders:
                    valid_model(model, _res_tag, dataloader=_hd_loader)
                    if args.ema:
                        ema.apply_shadow()
                        valid_model(model, f"ema_{_res_tag}", dataloader=_hd_loader)
                        ema.restore()
                _initial_eval_done = True   # [NEW] initial eval 완료 표시 (이후엔 eval_steps 주기로만)

            # Checkpoint
            # [FIX - oliviaa] _is_accum_step 게이팅 추가
            # optimizer_step 은 _accum 개의 연속된 current_step 동안 동일 값 유지 → 게이팅 없으면
            # save_ckpt_step 경계에서 _accum 번 연속 저장됨 (예: accum=4 → checkpoint-N0,N1,N2,N3 4개 동일 파일).
            if _is_accum_step and optimizer_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "student_patchify": student_patchify.state_dict() if student_patchify is not None else {},
                        "dics_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.state_dict() if args.ema else {},
                    # [NEW] LoRA weight save (dit_pipe.dit 의 lora_ param 만)
                    lora_state_dict=(
                        {n: p.data.cpu().clone()
                         for n, p in dit.named_parameters() if 'lora_' in n}
                        if getattr(args, 'use_lora', False) else {}
                    ),
                    optimizer_step=optimizer_step,  # accum 변경 시 wandb step 일관성 유지
                    grad_accum_steps=_accum,
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help="number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of epochs to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--resume_ckpt_grad_accum", type=int, default=None,
                        help="resume 시 ckpt 의 grad_accum_steps metadata 없을 때 사용 (= 옛 ckpt 호환)")
    parser.add_argument("--ckpt_dir", type=str, default="./results/", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="linear LR warmup: ramp 0->target over N optimizer steps from this run's start (0=off). resume 시에도 run 시작 기준.")
    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--no_log_grad", action="store_true",
                        help="disable gradient norm logging and retain_grad (saves GPU memory)")
    # [Modified - oliviaa] 원본: OSP에서 encoder만 freeze하는 옵션. 현재는 --freeze_pretrained으로 대체.
    # 필요하면 나중에 재활용 가능하므로 남겨둠.
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="[Stage 1.5] decoder 전체 freeze. student_patchify만 학습할 때 사용.")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="")

    # Data
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--num_frames", type=int, default=17, help="")
    # [NEW - mixed-length / variant B] 17/81 혼합 학습 (align 유지 + decoder 81f). 끄면 기존 고정길이.
    parser.add_argument("--mix_frames", action="store_true", default=False,
                        help="배치마다 확률로 17/81 프레임 섞어 학습 (MixedLengthBatchSampler)")
    parser.add_argument("--mix_81_prob", type=float, default=0.2,
                        help="81프레임 배치가 뽑힐 확률")
    parser.add_argument("--mix_81_num_frames", type=int, default=81,
                        help="혼합에 섞을 긴 길이")
    parser.add_argument("--mix_81_batch_size", type=int, default=2,
                        help="81프레임 전용 배치 크기 (align@81f bs2=157GB 측정완료)")
    parser.add_argument("--resolution", type=int, default=256, help="")
    parser.add_argument("--sample_rate", type=int, default=2, help="")
    parser.add_argument("--dynamic_sample", action="store_true", help="")
    # Generator model
    # [Removed - oliviaa] --ignore_mismatched_sizes: diffusers from_pretrained 전용 옵션, _video_vae에서 불필요
    # [Removed - oliviaa] --model_name: OSP ModelRegistry 전용, Wan VAE는 _video_vae()로 직접 로드
    # [Removed - oliviaa] --not_resume_training_process: 사용처 없음
    # [Removed - oliviaa] --model_config: OSP from_config 전용
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    # [NEW - oliviaa] latent 채널 수. pretrained=16. 변경 시 z_dim 관련 layer가 재초기화됨
    parser.add_argument("--z_dim", type=int, default=16, help="latent channel dim. pretrained=16")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help="path to Wan2.1_VAE.pth"
    )
    # [NEW - oliviaa] Added stages config — JSON string
    # 예: '[{"mode":"downsample3d","num_res_blocks":2}]'
    parser.add_argument("--add_encoder_stages", type=str, default=None,
                        help='JSON list of encoder stages, e.g. \'[{"mode":"downsample3d","num_res_blocks":2}]\'')
    parser.add_argument("--add_decoder_stages", type=str, default=None,
                        help='JSON list of decoder stages, e.g. \'[{"mode":"upsample3d","num_res_blocks":2}]\'')
    # [NEW - oliviaa] Freeze pretrained Wan VAE weights, train only added stages
    parser.add_argument("--freeze_pretrained", action="store_true", help="")
    # [NEW - oliviaa] freeze_pretrained 시 encoder 전체를 풀기 (add_downsamples만 푸는 기본 동작 대신)
    parser.add_argument("--unfreeze_encoder", action="store_true", help="")
    # [NEW - oliviaa] freeze_pretrained 시 decoder 전체를 풀기 (add_upsamples만 푸는 기본 동작 대신)
    parser.add_argument("--unfreeze_decoder", action="store_true", help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="")
    parser.add_argument("--wavelet_weight", type=float, default=0.1, help="")
    # Discriminator Model
    # [Removed - oliviaa] --load_disc_from_checkpoint: 현재 미사용
    # [Removed - oliviaa] --disc_cls: resolve_str_to_obj 전용, LPIPSWithDiscriminator3D 직접 호출로 변경
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")
    # [NEW - oliviaa] GAN adaptive weight 계산에 사용할 last_layer 선택
    # decoder_head: 원본 방식 (decoder.head[-1].weight). freeze_pretrained 시 사용 불가
    # add_upsamples: 추가된 decoder stage의 마지막 conv weight. freeze_pretrained 시 사용
    parser.add_argument("--gan_last_layer", type=str, default="decoder_head",
                        choices=["decoder_head", "add_upsamples"],
                        help="layer for GAN adaptive weight calculation")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_video_path", type=str, default=None, help="")
    parser.add_argument("--eval_num_frames", type=int, default=17, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_sample_rate", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    parser.add_argument("--eval_hd_subset_size", type=int, default=4,
                        help="HD eval(480x832x81 등) 전용 subset 크기 (base eval과 분리 — base 비교성 유지, HD는 적게=빠름)")
    # [NEW - oliviaa] Additional HD eval resolutions, comma-separated e.g. "512x512,480x832"
    parser.add_argument("--eval_resolutions_hd", type=str, default=None, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=2, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")
    parser.add_argument("--eval_noise_robust", action="store_true",
                        help="validation 때 decoder noise robustness(sigma sweep PSNR) 측정")
    parser.add_argument("--noise_robust_sigmas", type=str, default="0.1,0.2,0.3,0.5",
                        help="noise robustness sigma 목록 (z_main std 배수)")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="")

    # Wandb (replaces TensorBoard)
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="wandb run id to resume (e.g. '3rtlyr9o'). Empty = new run.")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    # [NEW - oliviaa/dit_align] DiT alignment args
    parser.add_argument("--dit_ckpt_dir", type=str,
                        default="checkpoints/Wan2.1-I2V-14B-480P",
                        help="DiT checkpoint directory (Wan2.1-I2V-14B). "
                             "Relative to repo root by default; pass absolute path to override.")
    parser.add_argument("--align_weight", type=float, default=0.5,
                        help="alignment loss weight (0 = disable)")
    parser.add_argument("--align_loss_type", type=str, default="l2_mean",
                        choices=["mse", "cosine", "l2_mean"],
                        help="alignment loss type. l2_mean: per-token L2 norm mean (g_loss와 스케일 유사)")
    parser.add_argument("--align_layers", type=str, default="all",
                        help="DiT layers for alignment. 'all' or comma-separated indices e.g. '0,10,20,30,39'")
    parser.add_argument("--align_agg", type=str, default="sum",
                        choices=["sum", "mean"], help="layer별 loss aggregation. sum: layer 합, mean: layer 평균")
    parser.add_argument("--patchify_lr", type=float, default=1e-4,
                        help="student patchify learning rate (separate from VAE lr)")
    parser.add_argument("--patchify_init", type=str, default="zero",
                        choices=["zero", "normal", "kaiming"],
                        help="추가 채널 초기화. zero: pretrained 동작 유지, normal: N(0,0.02), kaiming: Conv default")
    parser.add_argument("--patchify_mask_init", type=str, default="zero",
                        choices=["zero", "copy4_zero4", "copy4_half4"],
                        help="mask_main 8ch 초기화. zero: 전부 zero, copy4_zero4: 앞 4ch pretrained + 뒤 4ch zero, copy4_half4: 복제 × 0.5")
    parser.add_argument("--mask_mode", type=str, default="single8",
                        choices=["single8", "dual12"],
                        help="mask 채널 구성. single8: 8ch(tf=8), dual12: 8ch(tf=8)+4ch(tf=4, pretrained 호환)")
    parser.add_argument("--dit_dit_offload", action="store_true",
                        help="DiT blocks를 CPU에 offload. forward/backward 시에만 GPU로 이동. ~27GB VRAM 절약")
    parser.add_argument("--dit_fsdp2", action="store_true",
                        help="FSDP v2 (fully_shard)로 DiT blocks를 GPU 간 shard. CPU offload보다 빠름")
    parser.add_argument("--text_fsdp2", action="store_true",
                        help="FSDP v2로 UMT5 blocks+token_embedding 및 CLIP transformer를 shard. "
                             "UMT5 ~11.3GB→~2.8GB/GPU, CLIP ~1.3GB→~0.3GB/GPU. t5_offload와 상호 배타적.")
    parser.add_argument("--dit_num_inference_steps", type=int, default=50,
                        help="FlowMatch scheduler num steps (for noise schedule)")
    parser.add_argument("--dit_timestep_mode", type=str, default="random",
                        choices=["random", "fixed"], help="timestep sampling mode")
    parser.add_argument("--dit_fixed_timesteps", type=str, default="0,12,25,37,49",
                        help="fixed timestep indices (used when dit_timestep_mode=fixed)")
    parser.add_argument("--align_after_patchify", action="store_true", default=True,
                        help="patchify 직후 (block 0 입력)에도 alignment loss 적용")
    parser.add_argument("--no_align_after_patchify", dest="align_after_patchify", action="store_false",
                        help="patchify 직후 alignment 비활성화")
    parser.add_argument("--align_adaptive_weight", action="store_true", default=True,
                        help="adaptive weight 사용 (gradient ratio 로 스케일 자동 조정)")
    parser.add_argument("--no_align_adaptive_weight", dest="align_adaptive_weight", action="store_false",
                        help="adaptive weight 비활성화 (align_weight 를 직접 스케일로 사용)")
    # [NEW] Legacy 2-backward adaptive (autograd.grad x 2 + retain_graph). Mutually
    # exclusive with the default single-backward _AdaptiveWeightingFn path.
    # When True: forward() skips _AdaptiveWeightingFn.apply, train loop computes
    # w from gradients on encoder.head[-1].weight via compute_adaptive_weight_2bwd().
    # Requires --align_adaptive_weight (the master switch).
    parser.add_argument("--use_2backward_adaptive", action="store_true", default=False,
                        help="legacy 2-backward adaptive path (autograd.grad x2 + retain_graph). "
                             "Mutually exclusive with single-backward _AdaptiveWeightingFn. "
                             "Requires --align_adaptive_weight. Slower (~+20-30%% step time) but "
                             "matches encoder.head[-1].weight measurement of original production.")
    parser.add_argument("--adaptive_max_weight", type=float, default=1e4,
                        help="upper clamp for adaptive weight ratio (applies to both single-bwd and 2-bwd modes). "
                             "0 or negative = no clamp.")
    # [NEW - oliviaa/B-fix] (B) 식 — encoder.head[-1] 의 conv forward 의 분기 + weight gradient ratio
    parser.add_argument("--use_b_adaptive", action="store_true", default=False,
                        help="(B) 식 — encoder.head[-1] 자체 를 AdaptiveWeightedCausalConv3d 으로 교체. "
                             "single backward path 의 weight gradient ratio mechanism (= 정확 weight ratio). "
                             "use_2backward_adaptive 와 mutually exclusive. 기존 activation gradient ratio "
                             "(= _AdaptiveWeightingFn.apply) 대체.")
    parser.add_argument("--log_adaptive_weight", action="store_true",
                        help="wandb log adaptive weight w (raw ratio) + properly-weighted train/align_loss_weighted. "
                             "Default off (avoids per-step .item() cpu sync).")
    parser.add_argument("--no_fused_align", action="store_true",
                        help="origin (kk4aiuyq) 식 alignment 사용: "
                             "run_teacher → run_student → compute_alignment_loss (sum(grad-tracked losses)). "
                             "Default off = fused_dit_align_forward (per-block detach + AlignGradInjector inject, memory 효율). "
                             "True = origin 식 (teacher 의 모든 features 보존, peak memory ↑).")
    # [NEW] LoRA on student DiT (dit_training-compatible setting).
    parser.add_argument("--use_lora", action="store_true",
                        help="Inject LoRA into DiT body (student-side trainable). DiT body weights stay frozen; "
                             "only LoRA params trainable. teacher path disables adapter at runtime.")
    parser.add_argument("--freeze_lora", action="store_true",
                        help="[RESTORE] LoRA 를 frozen(requires_grad=False) 으로 → optimizer 미포함. "
                             "1x52jvfx(loraFreeze) 재현. (zero-init LoRA 라 --use_lora 생략과 기능 동일.)")
    parser.add_argument("--lora_rank", type=int, default=512,
                        help="LoRA rank (and alpha, equal). dit_training default = 512.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,k_img,v_img,ffn.0,ffn.2",
                        help="comma-separated module names to inject LoRA into. dit_training default.")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="optional safetensors LoRA ckpt to load on inject (resume). None = random init.")
    parser.add_argument("--t5_offload", action="store_true",
                        help="null text 계산 후 T5 를 CPU 로 offload. ~10GB VRAM 절약 (caption mode에서는 무시)")
    parser.add_argument("--normalize_zprior", action="store_true",
                        help="z_prior를 pretrained 통계로 normalize. teacher z_ref와 scale 일치.")
    # [NEW] REPA-E style: z_main 도 BN3d 로 online 정규화
    parser.add_argument("--normalize_zmain_bn", action="store_true",
                        help="z_main 을 BatchNorm3d 로 online 정규화 (REPA-E style). running stats EMA update.")
    parser.add_argument("--bn_momentum", type=float, default=0.1,
                        help="BN 의 EMA momentum (PyTorch / REPA-E default 0.1)")
    parser.add_argument("--zmain_bn_init", type=str, default="zprior",
                        choices=["zprior", "cold", "pytorch_default"],
                        help="zprior=z_prior pre_stats 로 init (REPA-E init_bn 동등), cold=0/1, pytorch_default=BN 기본")
    parser.add_argument("--freeze_patchify_zprior", action="store_true",
                        help="student patchify의 z_prior weight 고정 (pretrained copy 유지, z_main만 학습)")
    parser.add_argument("--freeze_patchify_full", action="store_true",
                        help="student patchify 전체 freeze (LoRA-only pure isolation 실험용)")
    # [NEW - oliviaa] RAE-style decoder noise augmentation
    parser.add_argument("--decoder_noise_tau_main", type=float, default=0.0,
                        help="z_main decoder noise: per-sample sigma ~ Uniform[0, tau]. "
                             "Applied in normalized space. Default 0 = no noise. "
                             "Measured DiT inference noise std max ≈ 1.54 → recommend tau ≈ 1.8")
    parser.add_argument("--decoder_noise_tau_prior", type=float, default=0.0,
                        help="z_prior decoder noise: per-sample sigma ~ Uniform[0, tau]. "
                             "Measured DiT inference noise std max ≈ 0.95 → recommend tau ≈ 1.1")
    parser.add_argument("--decoder_noise_random_mode", action="store_true", default=True,
                        help="Per-sample random mode: main_only / prior_only / both (1/3 each). "
                             "Default True. Disable with --no_decoder_noise_random_mode")
    parser.add_argument("--no_decoder_noise_random_mode", dest="decoder_noise_random_mode",
                        action="store_false",
                        help="Always apply noise on both channels (no random mode selection)")
    parser.add_argument("--decoder_noise_warmup_steps", type=int, default=0,
                        help="Curriculum: ramp tau from 0 to final value over this many "
                             "forward steps. Use to avoid cold-start shock when resuming from "
                             "clean-trained decoder. 0 = no warmup (apply full tau immediately).")
    parser.add_argument("--decoder_noise_warmup_power", type=float, default=1.0,
                        help="Power for curriculum schedule: warmup_factor = (step/total)^power. "
                             "1.0 = linear (default). 2.0 = quadratic (slow start, fast end). "
                             "3.0 = cubic (very slow start). 0.5 = sqrt (fast start, slow end).")
    # [NEW - oliviaa] Stage 1.5 (변종 B) — z_main을 precomputed stats로 정규화 후 alignment 학습.
    # 기본값 OFF → Stage 1 (기존) 동작과 동일.
    # ON 시 student_patchify가 normalized z_main에 calibration → Stage 2에서 같은 정규화 적용 시 직접 reuse 가능.
    parser.add_argument("--normalize_zmain", action="store_true",
                        help="[Stage 1.5 변종 B] z_main을 zmain_stats로 정규화해서 alignment에 흘림")
    parser.add_argument("--zmain_stats_path", type=str, default=None,
                        help="[Stage 1.5] z_main 통계 JSON 파일 경로 (--normalize_zmain와 함께 사용)")
    parser.add_argument("--caption_metadata", type=str, default=None,
                        help="caption jsonl 경로 (video→prompt). 지정 시 per-batch text encoding, T5 offload 불가")
    parser.add_argument("--t5_cache_dir", type=str, default=None,
                        help="pre-cached T5 embeddings 디렉토리. 지정 시 T5 없이 학습, offload 가능")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="gradient accumulation steps. effective_batch = batch_size * num_gpu * accum_steps")
    parser.add_argument("--use_grad_checkpoint", action="store_true",
                        help="student DiT branch 에 gradient checkpointing 적용")
    parser.add_argument("--grad_checkpoint_num_blocks", type=int, default=40,
                        help="gradient checkpoint 적용할 block 수 (앞에서부터). 40=전부, 15=앞쪽 15개만")
    parser.add_argument("--align_num_blocks", type=int, default=40,
                        help="alignment에 사용할 DiT block 수. 40=전부, 20=절반. 줄이면 activation 메모리 절약")
    # [NEW - oliviaa] block 별 학습 가능 projection layer (trilinear 후 residual)
    parser.add_argument("--use_align_projection", action="store_true", default=False,
                        help="trilinear 후 block 별 conv3d residual projection (zero init)")
    parser.add_argument("--align_proj_dim", type=int, default=5120,
                        help="projection conv 의 in/out channel (= DiT hidden_size)")
    parser.add_argument("--align_proj_bottleneck_dim", type=int, default=16,
                        help="bottleneck dim (= D→mid→D). 1x1 conv. mid=16 → ~164K params/block (40 block → 6.6M total). "
                             "mid=8 → 3.3M total. REPA/iREPA 표준 patterns.")
    parser.add_argument("--align_projection_init", type=str, default="zero", choices=["zero", "random"],
                        help="zero=trilinear 그대로 시작, random=Kaiming init")
    # [NEW - oliviaa] teacher noise 의 nearest subsample 으로 student noise 생성 (= correlated noise).
    # mode 별 의미:
    #   off     = both random (= 기존 동작, teacher correlation 없음)
    #   z_main  = (A) z_main = teacher subsample, z_prior = randn
    #   z_prior = (C) z_main = randn, z_prior = teacher subsample (= default, 학습 초기 align 효과 강함)
    #   both    = (B) z_main + z_prior 둘 다 같은 teacher subsample (= channel correlation 1, train/test mismatch 위험)
    parser.add_argument("--teacher_subsample_noise_mode", type=str, default='off',
                        choices=['off', 'z_main', 'z_prior', 'both'],
                        help="teacher noise subsample 적용 부분. C='z_prior' 권장")
    # [DEPRECATED] backward compat: 기존 flag 켜져 있으면 mode='z_prior' 로 자동 변환
    parser.add_argument("--use_teacher_subsample_noise", action="store_true", default=False,
                        help="[deprecated] 켜지면 --teacher_subsample_noise_mode=z_prior 로 변환")
    # [NEW - oliviaa] diffusion (flow matching) loss — REPA-E 방식 (z.detach()), stage2 rm57nmln 동일
    parser.add_argument("--use_diffusion_loss", action="store_true", default=False,
                        help="student DiT 에 flow matching diffusion loss 추가 (z_cat detach → VAE 무관). no_fused_align 필요")
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0,
                        help="diffusion loss 가중치 (total_loss += weight * loss_diff). REPA: diffusion 1.0 main")
    parser.add_argument("--diffusion_max_timestep_boundary", type=float, default=1.0,
                        help="diffusion timestep 상한 (0~1, scheduler.timesteps 비율). stage2 동일")
    parser.add_argument("--diffusion_min_timestep_boundary", type=float, default=0.0,
                        help="diffusion timestep 하한 (0~1)")
    # [RESTORE matchS2] align noise/weight 를 stage2 diffusion 학습과 동일하게 (1000-step training sched + per-sample t + bsmntw weight)
    parser.add_argument("--align_match_stage2", action="store_true", default=False,
                        help="[RESTORE] align 을 stage2 와 동일: 1000-step training scheduler + per-sample timestep "
                             "+ bsmntw(training_weight). default=off → 50-step shared-t (기존).")
    parser.add_argument("--align_max_timestep_boundary", type=float, default=1.0,
                        help="[RESTORE] matchS2 align timestep 상한 (0~1, 1000-step scheduler 비율). stage2 동일")
    parser.add_argument("--align_min_timestep_boundary", type=float, default=0.0,
                        help="[RESTORE] matchS2 align timestep 하한 (0~1)")
    parser.add_argument("--diffusion_unfreeze_zprior_patchify", action="store_true", default=False,
                        help="z_prior patchify freeze hook 끔 → diffusion+align 으로 학습 (stage2 rm57nmln 동일)")
    parser.add_argument("--diffusion_unfreeze_block_norms_mod", action="store_true", default=False,
                        help="DiT block norm+modulation unfreeze (~2.6M, rm57nmln unfreeze_block_norms_mod 동일)")
    # [NEW - oliviaa] ablation: align loss 의 grad 기여를 0 으로 → diffusion only.
    # align forward 는 v43 과 동일하게 그대로 실행 (B-adaptive dual-output graph topology 유지,
    # static_graph 안전). align_loss 만 ×0 → VAE/block_norms/patchify/projection 으로 가는 align grad 차단.
    # 목적: z_prior diffusion loss spike 가 align 충돌 때문인지 격리 검증 (align 없으면 0.19 근처 머무는지).
    parser.add_argument("--diffusion_only", action="store_true", default=False,
                        help="[ablation] align_loss 를 total 에 더할 때 ×0 (align grad 차단, diffusion 만 학습). DiT/davae_head 는 정상 로드")
    # [NEW - oliviaa] REPA-E 정석: align forward 동안 DiT(denoiser) freeze → align grad 가 VAE encoder 로만.
    # REPA-E train_repae.py:371-372 `requires_grad(SiT, False)` 동일. diffusion forward 는 정상 학습(z.detach).
    # 목적: align(VAE 정렬) 유지하면서 denoiser 충돌(z_prior spike) 제거. = alignment→VAE only / diffusion→denoiser only.
    parser.add_argument("--align_stop_grad_dit", action="store_true", default=False,
                        help="[REPA-E 정석] align forward 동안 student_patchify+LoRA+block_norms freeze → align grad 가 VAE encoder 로만 흐름. diffusion forward 는 정상")
    # [NEW] teacher forward(align target)를 init 시점 pretrained Wan I2V 14B 로 고정.
    # teacher/student 는 dit 객체를 공유 → student diffusion 이 LoRA/block_norms 를 학습하면 teacher align target 도 같이 drift 됨.
    # 이 플래그 on 이면 teacher forward 그 순간에만 (a) LoRA adapter off + (b) block_norms/mod 를 pretrained 초기값으로 swap → align target 이 init Wan 으로 고정. 직후 원복.
    # 별도 dit copy 불필요 (메모리 ≈ block_norms snapshot ~5MB). student/diffusion forward 는 학습된 weight 그대로.
    parser.add_argument("--teacher_frozen_pretrained", action="store_true", default=False,
                        help="[NEW] teacher forward 동안만 LoRA off + block_norms/mod 를 pretrained 초기값으로 swap → align target 을 init Wan I2V 14B 로 고정 (drift 차단)")
    # [NEW - oliviaa] teacher/student 공유 DiT 의 LoRA 를 baseline(256 finetuned) LoRA safetensors 로 init.
    # base Wan 은 동일 pretrained, LoRA 만 256 적응본으로 교체 → teacher align target + student denoiser 둘 다 256 친화 시작점.
    parser.add_argument("--init_lora_safetensors", type=str, default=None,
                        help="[NEW] DiT LoRA 초기값을 이 safetensors(PEFT 형식 lora_A/B) 로 로드. random init 대신 baseline LoRA 주입")
    parser.add_argument("--align_block_stride", type=int, default=1,
                        help="block 선택 간격. 1=연속(앞쪽 N개), 2=짝수번째(0,2,4,...). stride>1이면 전체 depth 커버")
    parser.add_argument("--align_batch_size", type=int, default=0,
                        help="alignment에 사용할 batch 크기. 0=전체 batch 사용, 1=첫 샘플만. rec은 전체 batch 유지")
    parser.add_argument("--lpips_chunk_size", type=int, default=0,
                        help="LPIPS를 chunk 단위로 순차 계산. 0=전체 한번에, >0=chunk size. batch_size>1일 때 메모리 절약")
    parser.add_argument("--max_grad_norm", type=float, default=0.0,
                        help="gradient clipping max norm. 0=비활성화, >0=clipping 적용. 7.0 권장")
    # [NEW - jeeyoung] per-component grad clip (decoder/encoder 따로). 0=off.
    parser.add_argument("--decoder_grad_clip", type=float, default=0.0,
                        help="decoder param gradient norm clip (0=off)")
    parser.add_argument("--encoder_head_grad_clip", type=float, default=0.0,
                        help="encoder.head[-1] gradient norm clip (0=off)")
    parser.add_argument("--encoder_body_grad_clip", type=float, default=0.0,
                        help="encoder body(head[-1] 제외) gradient norm clip (0=off)")
    parser.add_argument("--align_projections_grad_clip", type=float, default=0.0,
                        help="align_projections per-module gradient norm clip (0=off)")
    parser.add_argument("--student_patchify_grad_clip", type=float, default=0.0,
                        help="student_patchify per-module gradient norm clip (0=off)")

    args = parser.parse_args()

    # [NEW] backward compat: deprecated flag → new mode
    if getattr(args, 'use_teacher_subsample_noise', False) and args.teacher_subsample_noise_mode == 'off':
        args.teacher_subsample_noise_mode = 'z_prior'

    # [NEW] Mutual exclusion for adaptive weighting paths.
    if getattr(args, 'use_2backward_adaptive', False):
        assert args.align_adaptive_weight, (
            "--use_2backward_adaptive requires --align_adaptive_weight (master switch). "
            "Use --no_align_adaptive_weight to disable adaptive entirely instead.")
        assert args.align_weight > 0, (
            "--use_2backward_adaptive is meaningless when --align_weight 0.")

    set_random_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
