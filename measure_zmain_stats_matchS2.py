"""
matchS2 최신 ckpt(checkpoint-4000)의 EMA VAE로 z_main stats (mean/std) 측정.
- 영상 200개 (지금까지 최대), 256x256x17 (= 학습 setting).
- EMA weights(shadow) + raw z_main(mu, BN 안 거침) per-channel mean/std.
- stage2 zmain_stats json 포맷으로 저장 → --normalize_zmain 에 바로 사용 가능.
"""
import argparse, sys, json as _json
from functools import partial
import torch

CKPT = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/kinemadae_stage1_bn_lora_align40_bs8_clamp1e5_loraBN_matchS2_b_proj_teacherfrozen-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-4000.ckpt"
WAN_CKPT = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
VIDEO_EVAL = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt"
N = 200
OUT = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/zmain_stats_matchS2_step4000_256x256x17_n200.json"
DEVICE = "cuda:0"

# ── geoprior _video_vae 패치 (matchS2 launch args 동일) ──
import kinemadae_geoprior as kinemadae
kinemadae._video_vae = partial(
    kinemadae._video_vae_geoprior,
    dual_branch=True, subsample_mode="bilinear", prior_z_dim=16,
    add_decoder_tail_stages=None,
    add_decoder_before_head_stages=_json.loads('[{"mode":"upsample3d","num_res_blocks":2}]'),
    decoder_conv1_zmain_init="zero", expand_conv2=False, expand_encoder_head=False,
    use_b_adaptive=True,
)
sys.modules["kinemadae"] = kinemadae
from kinemadae import _video_vae
from video_dataset import ValidVideoDataset

print("[1] VAE 빌드...", flush=True)
vae = _video_vae(pretrained_path=WAN_CKPT, z_dim=16, device="cpu",
                 add_encoder_stages=_json.loads('[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]'),
                 add_decoder_stages=None).to(DEVICE).eval()

print("[2] EMA(shadow) weights 로드...", flush=True)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
shadow = sd["ema_state_dict"]["shadow"]   # 'module.vae.encoder...'
stripped = {}
for k, v in shadow.items():
    nk = k[len("module.vae."):] if k.startswith("module.vae.") else k
    stripped[nk] = v
vae_keys = set(vae.state_dict().keys())
use = {k: v for k, v in stripped.items() if k in vae_keys}
vae.load_state_dict(use, strict=False)
print(f"    overlay {len(use)} keys (EMA encoder)", flush=True)

print(f"[3] val clips {N}개 로드 + encode (mu)...", flush=True)
ds = ValidVideoDataset(real_video_dir=VIDEO_EVAL, num_frames=17, sample_rate=1, crop_size=256, resolution=256)
import torch as T
sums = None; sqs = None; cnt = 0; nvid = 0; i = 0
with T.no_grad():
    while nvid < N and i < len(ds):
        try:
            x = ds[i]["video"].unsqueeze(0).to(DEVICE)  # (1,C,T,H,W)
        except Exception as e:
            i += 1; continue
        er = vae.encode(x, scale=None)
        if isinstance(er, tuple) and len(er) == 2 and isinstance(er[0], tuple):
            (mu, _), _ = er
        else:
            mu, _ = er
        # mu: (1, 16, T', H', W') → per-channel 누적
        m = mu.float()
        C = m.shape[1]
        flat = m.permute(1, 0, 2, 3, 4).reshape(C, -1)  # (16, N)
        if sums is None:
            sums = flat.sum(dim=1); sqs = (flat**2).sum(dim=1); cnt = flat.shape[1]
        else:
            sums += flat.sum(dim=1); sqs += (flat**2).sum(dim=1); cnt += flat.shape[1]
        nvid += 1; i += 1
        if nvid % 25 == 0:
            print(f"    {nvid}/{N} videos", flush=True)

mean = (sums / cnt)
var = (sqs / cnt) - mean**2
std = var.clamp_min(1e-12).sqrt()
print(f"[4] 측정 완료: {nvid} videos, {cnt} latent positions/channel", flush=True)
print(f"    mean (16ch): min={mean.min():.4f} max={mean.max():.4f} avg={mean.mean():.4f}", flush=True)
print(f"    std  (16ch): min={std.min():.4f} max={std.max():.4f} avg={std.mean():.4f}", flush=True)
out = {
    "mean": mean.tolist(), "std": std.tolist(), "inv_std": (1.0/std).tolist(),
    "n_samples": nvid, "source": "matchS2 checkpoint-4000.ckpt (EMA, raw z_main mu)",
    "z_dim": 16, "num_frames": 17, "resolution": [256, 256],
}
with open(OUT, "w") as f:
    _json.dump(out, f, indent=2)
print(f"[5] 저장: {OUT}", flush=True)
