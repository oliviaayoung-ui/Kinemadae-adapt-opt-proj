"""dataset 의 sample 으로 z_main 의 실제 mean / std 측정 + wyy6dfgz BN stats 와 비교.

목적:
  - wyy6dfgz step-26000 ckpt 의 zmain_bn.running_mean / running_var 와
    dataset 의 실제 z_main 의 mean / std 의 일치 검증.
  - BN stats 가 정확 한 dataset 통계 의 추정 인지 확인.

GPU 사용 — 단 1개 GPU 의 VAE encode (= 작은 메모리, eval mode).
"""
import sys, torch, os
import torch.nn.functional as F

# 1. dataset 의 sample 의 path
VIDEO_LIST = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt"
N_SAMPLES = 128
DEVICE = 'cuda:0'

with open(VIDEO_LIST) as f:
    video_paths = [line.strip() for line in f.readlines()][:N_SAMPLES]
print(f"sample {N_SAMPLES} video paths loaded")

# 2. wyy6dfgz step-26000 ckpt 의 VAE load
sys.path.insert(0, '/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-lora-bn-clamp1e7')
import kinemadae_geoprior as kg

ckpt_path = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-lora-bn-clamp1e7/results/kinemadae_lora_bn_clamp1e7_align40_bs4_resume1k_bs4ga4-lr8.00e-05-bs4-rs256-sr2-fr17/checkpoint-26000.ckpt"
print(f"loading ckpt: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
gen_sd = ckpt['state_dict']['gen_model']

# 3. VAE build (= stage1 의 setup 그대로)
vae = kg._video_vae_geoprior(
    pretrained_path=None,
    z_dim=16,
    device='cpu',
    add_encoder_stages=[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}],
    add_decoder_before_head_stages=[{"mode":"upsample3d","num_res_blocks":2}],
    dual_branch=True,
    subsample_mode='bilinear',
    prior_z_dim=16,
    expand_conv2=False,
)

# vae 만 의 state_dict load (= 'vae.' prefix)
vae_sd = {k[len('vae.'):]: v for k, v in gen_sd.items() if k.startswith('vae.')}
missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
print(f"vae load — missing: {len(missing)}, unexpected: {len(unexpected)}")
vae = vae.to(DEVICE).eval()

# 4. dataset 의 sample 의 video load + encode
print(f"\nencoding {N_SAMPLES} videos ...")
try:
    import decord
    decord.bridge.set_bridge('torch')
except ImportError:
    print("decord 없음 — pip install decord 필요")
    sys.exit(1)

all_z_main = []
with torch.no_grad():
    for i, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            print(f"  [skip] {vp} 없음")
            continue
        try:
            vr = decord.VideoReader(vp, num_threads=1)
            n_frames = min(17, len(vr))
            indices = list(range(0, n_frames))
            frames = vr.get_batch(indices)  # (T, H, W, 3) uint8
            frames = frames.permute(3, 0, 1, 2).float() / 127.5 - 1.0  # (3, T, H, W) [-1, 1]
            # resize to 256
            T = frames.shape[1]
            f = frames.permute(1, 0, 2, 3).reshape(T, 3, frames.shape[2], frames.shape[3])
            f = F.interpolate(f, size=(256, 256), mode='bilinear', align_corners=False)
            # center crop already (= 256x256)
            frames = f.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)  # (1, 3, T, H, W)
            mu, _ = vae.encode(frames, scale=None)
            all_z_main.append(mu.cpu())   # (1, 16, T', H', W')
            print(f"  [{i+1}/{N_SAMPLES}] {os.path.basename(vp)} → z_main shape {tuple(mu.shape)}")
        except Exception as e:
            print(f"  [err] {vp}: {str(e)[:80]}")

# 5. mean, std 측정 (= channel 별)
z_cat = torch.cat(all_z_main, dim=0)   # (N, 16, T', H', W')
print(f"\ntotal z_main shape: {tuple(z_cat.shape)}")
# channel 별 mean / std (= dim 0 + spatial)
z_mean = z_cat.mean(dim=(0, 2, 3, 4))   # (16,)
z_std  = z_cat.std (dim=(0, 2, 3, 4))   # (16,)

# 6. ckpt 의 BN stats
bn_mean = ckpt['ema_state_dict']['shadow_buffers']['module.zmain_bn.running_mean'].cpu()
bn_var  = ckpt['ema_state_dict']['shadow_buffers']['module.zmain_bn.running_var'].cpu()
bn_std  = bn_var.sqrt()

# 7. 비교
print(f"\n{'='*70}")
print(f"=== mean 비교 (channel 별) ===")
print(f"{'ch':>3} {'measured':>10} {'BN ckpt':>10} {'diff':>10}")
for c in range(16):
    diff = (z_mean[c] - bn_mean[c]).item()
    print(f"{c:>3} {z_mean[c].item():>10.4f} {bn_mean[c].item():>10.4f} {diff:>+10.4f}")
print(f"\nmean abs diff:")
print(f"  max:  {(z_mean - bn_mean).abs().max().item():.4f}")
print(f"  mean: {(z_mean - bn_mean).abs().mean().item():.4f}")

print(f"\n{'='*70}")
print(f"=== std 비교 (channel 별) ===")
print(f"{'ch':>3} {'measured':>10} {'BN ckpt':>10} {'ratio':>10}")
for c in range(16):
    ratio = (z_std[c] / bn_std[c]).item()
    print(f"{c:>3} {z_std[c].item():>10.4f} {bn_std[c].item():>10.4f} {ratio:>10.4f}")
print(f"\nstd ratio (measured / BN):")
print(f"  mean:  {(z_std / bn_std).mean().item():.4f}  (= 1.0 일수록 BN stats 정확)")
print(f"  range: [{(z_std / bn_std).min().item():.4f}, {(z_std / bn_std).max().item():.4f}]")
