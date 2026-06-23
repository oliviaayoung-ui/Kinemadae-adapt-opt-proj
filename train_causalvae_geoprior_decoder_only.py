# [decoder_only] Geoprior VAE decoder-only training (DiT alignment STRIPPED).
#
# Purpose: take an aligned latent space (z_main) and FIX it, training only the
# DECODER (+ discriminator) to improve reconstruction. Encoder is FROZEN.
# Derived from train_causalvae_geoprior_dit_align.py with all DiT/alignment code removed.
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


# [decoder_only] DiT/align imports removed — this is a pure VAE reconstruction script.

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
from ddp_sampler import CustomDistributedSampler
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
        # [decoder_only] align_projections removed (no DiT alignment).
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

        # [decoder_only] No DiT alignment → no adaptive gradient weighting / no z_cat split.
        # Pure VAE: the decoder branch uses z_cat directly.
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
                z_main_dec = self._denorm_zmain(z_main_rec)

            # z_prior
            if tau_prior_eff > 0 and self.normalize_zprior:
                sigma_p = tau_prior_eff * torch.rand(
                    (B, 1, 1, 1, 1), device=device, dtype=dtype) * apply_prior
                z_prior_dec = self._denorm_zprior(z_prior_rec + sigma_p * torch.randn_like(z_prior_rec))
            else:
                z_prior_dec = self._denorm_zprior(z_prior_rec)

            z_cat_raw = torch.cat([z_main_dec, z_prior_dec], dim=1)
        else:
            z_cat_raw = torch.cat([self._denorm_zmain(z_main_rec), self._denorm_zprior(z_prior_rec)], dim=1)
        recon = self.vae.decode(z_cat_raw, scale=None)
        return recon, mu, log_var, z_cat




# [decoder_only] _AdaptiveWeightingFn and compute_adaptive_weight_2bwd removed
# (only used by the stripped DiT alignment path).


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

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank)
            with torch.cuda.amp.autocast(dtype=precision):
                outputs = model(inputs)
                video_recon = outputs[0]

            # Upload videos
            if global_rank == 0:
                for i in range(len(video_recon)):
                    if num_video_log <= 0:
                        break
                    gt_video = tensor_to_video(inputs[i])
                    rec_video = tensor_to_video(video_recon[i])
                    # [FIX - jeeyoung] gt/recon (T,C,H,W) 프레임·H 불일치 방어 — 공통 길이로 crop 후 concat(axis=3=W).
                    #   stage1 recon 은 같은 입력이라 보통 동일하지만 stage2/align gen-snap 버그와 일관되게 방어.
                    _tt = min(gt_video.shape[0], rec_video.shape[0]); _hh = min(gt_video.shape[2], rec_video.shape[2])
                    gt_video = gt_video[:_tt, :, :_hh]; rec_video = rec_video[:_tt, :, :_hh]
                    concat_video = np.concatenate([gt_video, rec_video], axis=3)
                    video_log.append(concat_video)
                    num_video_log -= 1

            B, C, T, H, W = inputs.shape
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            video_recon = rearrange(
                video_recon, "b c t h w -> (b t) c h w"
            ).contiguous()

            # Calculate per-video PSNR (one value per video, not per batch)
            # to avoid partial-batch bias when DDP gather averages across ranks
            mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))  # (B*T,)
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
    return psnr_list, lpips_list, video_log


def gather_valid_result(psnr_list, lpips_list, video_log_list, rank, world_size):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_video_logs = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_video_logs, video_log_list)

    # [decoder_only] latent-drift metrics (CKNNA/CKA/cosine) removed.
    return (
        np.mean(list(chain(*gathered_psnr_list))),
        np.mean(list(chain(*gathered_lpips_list))) if any(gathered_lpips_list) else 0.0,
        list(chain(*gathered_video_logs)),
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

    # ─── [decoder_only] DiT pipeline + alignment fully removed ───
    # Pure VAE reconstruction training: no teacher/student DiT, no patchify,
    # no flow-match schedulers, no text context.
    dit_pipe = None
    dit = None
    student_patchify = None
    if global_rank == 0:
        logger.info("[decoder_only] DiT/align disabled — pure VAE reconstruction training.")

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
    # [decoder_only] DaVaeHead (diffusion loss) removed.
    model.davae_head = None

    # [decoder_only] FREEZE encoder + conv1 + conv2 → Decoder3d 만 학습 (modules_to_train=[decoder] 와 일치).
    #   z_main = reparameterize(conv1(encoder(x))) → encoder+conv1 얼려야 aligned latent 고정 (conv1 안 얼면 latent 바뀜=버그).
    #   conv2(post-quant z→decoder)도 freeze: GAN 토글이 modules_to_train=[decoder]만 관리하므로 conv2 trainable이면 비정합.
    #   = 기존 --freeze_encoder(encoder+conv1+conv2) 의미. z_prior path는 forward에서 no_grad. conv2_prior는 z_dim==16라 미생성.
    #   gen optimizer 가 filter(p.requires_grad) → decoder 만 학습. (optimizer 생성 전에 위치.)
    for p in model.vae.encoder.parameters():
        p.requires_grad_(False)
    model.vae.conv1.requires_grad_(False)   # quant conv (enc out → mu/log_var) — z_main 고정 핵심
    model.vae.conv2.requires_grad_(False)   # post-quant (z → decoder) — decoder만 학습
    if global_rank == 0:
        _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(f"[decoder_only] encoder+conv1 frozen (z_main 고정). Trainable: {_n_train:,} | "
                    f"Frozen: {_n_frozen:,}")

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
    dataloader = DataLoader(
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
            if 'x' in res_str:
                h, w = map(int, res_str.split('x'))
                res = (h, w)
                name_tag = f"{h}x{w}"
            else:
                res = int(res_str)
                name_tag = str(res)
            hd_bs = max(1, args.eval_batch_size // 4)  # 고해상도는 batch 줄임
            hd_dataset = ValidVideoDataset(
                real_video_dir=args.eval_video_path,
                num_frames=args.eval_num_frames,
                sample_rate=args.eval_sample_rate,
                crop_size=res,
                resolution=res,
            )
            hd_subset = Subset(hd_dataset, indices=range(args.eval_subset_size))
            hd_sampler = CustomDistributedSampler(hd_subset)
            hd_loader = DataLoader(hd_subset, batch_size=hd_bs, sampler=hd_sampler, pin_memory=True)
            hd_val_dataloaders.append((name_tag, hd_loader))

    # [decoder_only] Optimizer — VAE only (encoder frozen above → trains decoder).
    vae_module = model.module.vae

    # filter(p.requires_grad, ...) → encoder params (frozen above) excluded → decoder only.
    vae_params = [p for p in vae_module.parameters() if p.requires_grad]

    # modules_to_train: set_train/set_eval/set_modules_requires_grad 에서 사용 (decoder only).
    modules_to_train = [vae_module.decoder]

    param_groups = [{'params': vae_params, 'lr': args.lr}]
    gen_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
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
        # [decoder_only] no student_patchify to load.
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
    elif getattr(args, 'init_vae_from', ""):
        # [decoder_only] WEIGHTS-ONLY init from an aligned-VAE checkpoint.
        # Unlike --resume_from_checkpoint (full resume: optimizer + step + EMA + sampler),
        # this loads ONLY model/disc weights and starts a FRESH run (step 0, fresh optimizer,
        # fresh sampler, fresh EMA initialized from these loaded weights below).
        if not os.path.isfile(args.init_vae_from):
            raise Exception(f"Make sure `{args.init_vae_from}` is a ckpt file.")
        _ckpt = torch.load(args.init_vae_from, map_location="cpu")
        model.module.load_state_dict(_ckpt["state_dict"]["gen_model"], strict=False)
        if "dics_model" in _ckpt["state_dict"]:
            disc.module.load_state_dict(_ckpt["state_dict"]["dics_model"])
        if global_rank == 0:
            logger.info(
                f"[decoder_only] Weights-only init from {args.init_vae_from} "
                f"(gen_model + disc; fresh optimizer/step/EMA, start at step 0)."
            )
        del _ckpt

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
                   "total_loss": 0.0}
    # [FIX] optim step 초기화: resume 시 ckpt 의 optim_step 사용 (= accum 변경 시 일관성). fresh 시 0.
    optimizer_step = _resume_optim_step if args.resume_from_checkpoint and '_resume_optim_step' in dir() else 0

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

                # ─── [decoder_only] DiT alignment removed. Pure VAE reconstruction. ───
                # total loss = VAE reconstruction (g_loss) only.
                total_loss = g_loss

                # gradient accumulation 지원
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

                # [v27] grad norm log (= 두 run 동등성 비교용, K=10 step 마다, rank0 only)
                if global_rank == 0 and (current_step % 10 == 0):
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
                        _log = {
                            "gnorm/encoder_body":    _gn(_enc_body),
                            "gnorm/encoder_head":    _gn(_enc_head),
                            "gnorm/decoder":         _gn(_dec),
                        }
                        wandb.log(_log, step=optimizer_step)
                    except Exception as _ge:
                        if current_step < 5:
                            logger.warning(f"[gnorm] fail: {_ge}")

                # [decoder_only] B-fix verify block removed.

                # accumulation 완료 시에만 optimizer step
                if _is_accum_step:
                    # [decoder_only] No DDP-外 params (student_patchify/LoRA/block_norms removed)
                    # → DDP handles all gradient sync internally.

                    # [NEW - oliviaa] gradient clipping + norm 로깅
                    scaler.unscale_(gen_optimizer)
                    _max_grad_norm = getattr(args, 'max_grad_norm', 0.0)
                    if _max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()), _max_grad_norm)
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
                        wandb.log(_grad_log, step=optimizer_step)

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
                        _tl = _loss_accum["total_loss"]
                    else:
                        _gl = g_loss.item()
                        _rl = g_log['train/rec_loss']
                        _kl = g_log['train/kl_loss']
                        _nl = g_log['train/nll_loss']
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
                    wandb.log({"train/total_loss": _tl}, step=optimizer_step)
                    # [decoder_only] align/diffusion/adaptive-weight logs removed.

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
                psnr_list, lpips_list, video_log = valid(
                    global_rank, rank, model, _loader, precision, args,
                    lpips_model=shared_lpips_model,
                )
                valid_psnr, valid_lpips, valid_video_log = gather_valid_result(
                    psnr_list, lpips_list, video_log, rank, dist.get_world_size(),
                )

                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    # video: wandb.Video accepts (N, T, C, H, W) or (T, C, H, W).
                    # tensor_to_video already returns (T, C, H, W); stacked → (N, T, C, H, W). No transpose needed.
                    _vid_arr = np.array(valid_video_log)
                    wandb.log({f"val{name}/recon": wandb.Video(_vid_arr, fps=10)}, step=optimizer_step)
                    wandb.log({f"val{name}/psnr": valid_psnr}, step=optimizer_step)
                    wandb.log({f"val{name}/lpips": valid_lpips}, step=optimizer_step)
                    logger.info(f"{name} Validation done.")

            if _is_accum_step and args.eval_video_path is not None and (optimizer_step % args.eval_steps == 0 or optimizer_step == 1):
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
                        # [decoder_only] student_patchify removed → empty (resume compat).
                        "student_patchify": {},
                        "dics_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.state_dict() if args.ema else {},
                    # [decoder_only] no LoRA.
                    lora_state_dict={},
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
    # [decoder_only] Weights-only init from an aligned-VAE checkpoint.
    # Loads gen_model (+ disc) weights ONLY; starts a FRESH run (step 0, fresh
    # optimizer/scheduler/sampler/EMA). Unlike --resume_from_checkpoint (full resume).
    # If both are set, --resume_from_checkpoint takes precedence.
    parser.add_argument("--init_vae_from", type=str, default="",
                        help="[decoder_only] path to a .ckpt to load gen_model/disc weights from "
                             "(weights-only; fresh optimizer/step/EMA). Normally used instead of "
                             "--resume_from_checkpoint.")
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
    # [NEW - oliviaa] Additional HD eval resolutions, comma-separated e.g. "512x512,480x832"
    parser.add_argument("--eval_resolutions_hd", type=str, default=None, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=2, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")

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
