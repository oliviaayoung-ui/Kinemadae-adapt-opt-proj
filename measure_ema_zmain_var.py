"""
EMA weight로 실제 z_main 분산을 측정해서 EMA running_var(=1.5M)이랑 직접 비교.
- su0jsilv (BN, LoRA freeze, 붕괴) checkpoint-5000 사용.
- VAE-only forward (DiT/teacher 안 띄움). CPU 실행 (학습 GPU 안 건드림).
- EMA weight encode vs LIVE weight encode 둘 다 측정 → z_main 실제 분산.
- 비교: EMA running_var(shadow_buffers)=1.5M, LIVE running_var=0.58.
"""
import argparse, sys, json as _json
from functools import partial
import torch

CKPT = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/results/kinemadae_stage1_bn_lora_align40_bs4_1bwd_loraFreeze_b_proj_teacherfrozen-lr8.00e-05-bs4-rs256-sr2-fr17/checkpoint-5000.ckpt"
WAN_CKPT = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
VIDEO_EVAL = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt"
N_CLIPS = 6
DEVICE = "cpu"

# ── geoprior _video_vae 패치 (train script line 29-68 재현, su0jsilv launch args) ──
import kinemadae_geoprior as kinemadae
kinemadae._video_vae = partial(
    kinemadae._video_vae_geoprior,
    dual_branch=True,
    subsample_mode="bilinear",
    prior_z_dim=16,
    add_decoder_tail_stages=None,
    add_decoder_before_head_stages=_json.loads('[{"mode":"upsample3d","num_res_blocks":2}]'),
    decoder_conv1_zmain_init="zero",
    expand_conv2=False,          # --no_expand_conv2
    expand_encoder_head=False,
    use_b_adaptive=True,         # --use_b_adaptive
)
sys.modules["kinemadae"] = kinemadae
from kinemadae import _video_vae
from video_dataset import ValidVideoDataset

print("[1] VAE 빌드 (pretrained Wan + geoprior stages)...", flush=True)
vae = _video_vae(
    pretrained_path=WAN_CKPT,
    z_dim=16,
    device="cpu",
    add_encoder_stages=_json.loads('[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]'),
    add_decoder_stages=None,
).to(DEVICE).eval()

print("[2] checkpoint 로드 (EMA shadow + live gen_model)...", flush=True)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
shadow = sd["ema_state_dict"]["shadow"]          # 'module.vae.encoder...' (EMA trainable weights)
sbuf = sd["ema_state_dict"]["shadow_buffers"]
live_gm = sd["state_dict"]["gen_model"]           # 'vae.encoder...' / 'zmain_bn...'

ema_running_var = sbuf["module.zmain_bn.running_var"].float().mean().item()
live_running_var = [live_gm[k] for k in live_gm if "zmain_bn.running_var" in k][0].float().mean().item()


def strip(d, prefixes):
    out = {}
    for k, v in d.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out


def load_weights(tag, weight_dict, prefixes):
    # vae 에 weight overlay (frozen pretrained 는 그대로, trainable 만 덮어씀)
    stripped = strip(weight_dict, prefixes)
    vae_keys = set(vae.state_dict().keys())
    use = {k: v for k, v in stripped.items() if k in vae_keys}
    missing = vae.load_state_dict(use, strict=False)
    print(f"    [{tag}] overlay {len(use)} keys (vae 총 {len(vae_keys)})", flush=True)


print("[3] val clips 로드...", flush=True)
ds = ValidVideoDataset(real_video_dir=VIDEO_EVAL, num_frames=17, sample_rate=1,
                       crop_size=256, resolution=256)
clips = []
i = 0
while len(clips) < N_CLIPS and i < len(ds):
    try:
        clips.append(ds[i]["video"])
    except Exception as e:
        print(f"    skip {i}: {e}", flush=True)
    i += 1
x = torch.stack(clips, 0).to(DEVICE)   # (B,C,T,H,W)
print(f"    clips: {tuple(x.shape)}", flush=True)


@torch.no_grad()
def measure(tag, weight_dict, prefixes):
    load_weights(tag, weight_dict, prefixes)
    vae.eval()
    er = vae.encode(x, scale=None)
    # use_b_adaptive: 이중 tuple 가능 → 첫 branch
    if isinstance(er, tuple) and len(er) == 2 and isinstance(er[0], tuple):
        (mu, log_var), _ = er
    else:
        mu, log_var = er
    std = torch.exp(0.5 * log_var)
    z_main = mu + std * torch.randn_like(std)     # reparameterized (BN 이 보는 값)
    # per-channel var over (B,T,H,W) = BN3d 통계 방식
    var_mu = mu.float().var(dim=(0, 2, 3, 4)).mean().item()
    var_z = z_main.float().var(dim=(0, 2, 3, 4)).mean().item()
    print(f"    [{tag}] z_main 실제 분산: mu={var_mu:.4f}, reparam={var_z:.4f}", flush=True)
    return var_mu, var_z


print("[4] EMA weight forward...", flush=True)
ema_mu, ema_z = measure("EMA-weight", shadow, ["module.vae."])
print("[5] LIVE weight forward...", flush=True)
live_mu, live_z = measure("LIVE-weight", live_gm, ["vae."])

print("\n" + "=" * 64, flush=True)
print("결과: z_main 실제 분산 vs BN running_var (checkpoint-5000)", flush=True)
print("=" * 64, flush=True)
print(f"{'':22}{'실제 z_main var':>16}{'BN running_var':>18}{'배율':>12}", flush=True)
print(f"{'EMA weight / EMA run':22}{ema_z:>16.3f}{ema_running_var:>18.1f}{ema_running_var/ema_z:>12.0f}", flush=True)
print(f"{'LIVE weight / LIVE run':22}{live_z:>16.3f}{live_running_var:>18.3f}{live_running_var/live_z:>12.2f}", flush=True)
print("\n해석:", flush=True)
print(f"  - EMA weight 가 만드는 z_main 실제 분산 = {ema_z:.3f} (O(1))", flush=True)
print(f"  - 근데 val_ema 가 쓰는 EMA running_var = {ema_running_var:.0f}", flush=True)
print(f"  → EMA running 이 실제보다 {ema_running_var/ema_z:.0f}배 큼 = stats 가 weight 에 전혀 안 맞음", flush=True)
print(f"  - LIVE 는 weight({live_z:.3f}) vs running({live_running_var:.3f}) 거의 일치 → val 정상", flush=True)
