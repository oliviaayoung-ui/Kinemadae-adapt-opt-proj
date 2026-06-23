"""순수 Wan I2V pretrained VAE (teacher, 16ch) 의 latent 채널별 mean / std 측정.

= measure_zmain_stats.py 의 Wan-pretrained 버전.
  measure_zmain_stats.py : geoprior VAE (dual-branch z_main) @ 256
  이 스크립트            : 순수 Wan2.1 VAE (pipe.vae, teacher) @ 480×W

출력 latent = WanVideoVAE.encode → model.encode 가 (mu - mean)*(1/std) 적용한 SCALED latent.
  Wan 공식 mean/std (wan_video_vae.py:1063) 로 normalize 된 값이라, 이 dataset(panda70m) 이
  Wan 학습분포와 같으면 채널별 mean≈0 std≈1 이어야 함. 실제 측정값의 0/1 이탈 = 분포 차이.

GPU: VAE 만 로드 (14B DiT 안 띄움), single video 씩 tiled encode → 작은 메모리.
"""
import sys, os, torch
import torch.nn.functional as F

# ---- 설정 (절대경로) ----
VIDEO_LIST = os.environ.get("VIDEO_LIST", "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt")
TAG        = os.environ.get("TAG", "")   # 출력 파일명 구분 (예: kpcjuodt)
VAE_PTH    = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
DIFFSYNTH  = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio"
N_SAMPLES  = int(os.environ.get("N_SAMPLES", 128))
HEIGHT     = int(os.environ.get("HEIGHT", 480))
WIDTH      = int(os.environ.get("WIDTH", 832))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", 81))
DEVICE     = os.environ.get("DEVICE", "cuda:0")

sys.path.insert(0, DIFFSYNTH)
from diffsynth.models.wan_video_vae import WanVideoVAE

print(f"=== Wan I2V VAE stats @ {HEIGHT}x{WIDTH}x{NUM_FRAMES}, N={N_SAMPLES}, device={DEVICE} ===")

# 1. VAE 로드 (model 가중치만)
vae = WanVideoVAE(z_dim=16)
sd = torch.load(VAE_PTH, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
# Wan2.1_VAE.pth = VideoVAE_ state dict → vae.model 로 로드
missing, unexpected = vae.model.load_state_dict(sd, strict=False)
print(f"VAE load — missing={len(missing)} unexpected={len(unexpected)}")
if len(missing) > 5:
    print("  [warn] missing 많음, 일부:", missing[:5])
vae = vae.to(DEVICE).eval()

# 2. 비디오 로드 + encode
if VIDEO_LIST.endswith(".jsonl"):
    import json as _json
    video_paths = []
    with open(VIDEO_LIST) as f:
        for l in f:
            if len(video_paths) >= N_SAMPLES: break
            try: video_paths.append(_json.loads(l)["video"])
            except Exception: pass
else:
    with open(VIDEO_LIST) as f:
        video_paths = [l.strip() for l in f if l.strip()][:N_SAMPLES]
print(f"video paths: {len(video_paths)}  (src={VIDEO_LIST})")

try:
    import decord
    decord.bridge.set_bridge("torch")
except ImportError:
    print("decord 필요"); sys.exit(1)

all_z = []
done = 0
with torch.no_grad():
    for i, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            continue
        try:
            vr = decord.VideoReader(vp, num_threads=1)
            nf = min(NUM_FRAMES, len(vr))
            frames = vr.get_batch(list(range(nf)))          # (T,H,W,3) uint8
            frames = frames.permute(3, 0, 1, 2).float() / 127.5 - 1.0  # (3,T,H,W) [-1,1]
            T = frames.shape[1]
            f4 = frames.permute(1, 0, 2, 3)                  # (T,3,H,W)
            f4 = F.interpolate(f4, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
            vid = f4.permute(1, 0, 2, 3).to(DEVICE)          # (3,T,H,W)
            z = vae.encode([vid], device=DEVICE, tiled=True)[0]   # (16,T',H',W')  scaled mu
            all_z.append(z.float().cpu())
            done += 1
            if done % 16 == 0:
                print(f"  [{done}] {os.path.basename(vp)} → z {tuple(z.shape)}")
        except Exception as e:
            print(f"  [err] {os.path.basename(vp)}: {str(e)[:80]}")

print(f"\nencoded {done} videos")
# 3. 채널별 mean / std (spatial + temporal + sample 전체)
flat = torch.cat([z.reshape(16, -1) for z in all_z], dim=1)   # (16, N*T'*H'*W')
ch_mean = flat.mean(dim=1)
ch_std  = flat.std(dim=1)
ch_var  = flat.var(dim=1)

print(f"\ntotal tokens per channel: {flat.shape[1]}")
print(f"\n{'='*56}")
print(f"=== Wan I2V VAE latent 채널별 통계 @ {HEIGHT}x{WIDTH} ===")
print(f"{'ch':>3} {'mean':>10} {'std':>10} {'var':>10}")
for c in range(16):
    print(f"{c:>3} {ch_mean[c].item():>10.4f} {ch_std[c].item():>10.4f} {ch_var[c].item():>10.4f}")
print(f"\n전체: mean(|ch_mean|)={ch_mean.abs().mean():.4f}  mean(ch_std)={ch_std.mean():.4f}")
print(f"  (Wan scale 적용 후라 dataset 분포가 Wan 학습분포와 같으면 mean≈0, std≈1)")

# 4. 저장 (plot 용)
_suffix = f"_{TAG}" if TAG else ""
out_pt = f"/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/wan_vae_stats_{HEIGHT}x{WIDTH}{_suffix}.pt"
torch.save({"mean": ch_mean, "std": ch_std, "var": ch_var,
            "height": HEIGHT, "width": WIDTH, "num_frames": NUM_FRAMES,
            "n_videos": done, "n_tokens": flat.shape[1]}, out_pt)
print(f"\nsaved: {out_pt}")
