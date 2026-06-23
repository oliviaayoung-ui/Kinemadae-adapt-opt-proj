"""채널 PCA → top3 → RGB spatial 시각화 (DINO 스타일 feature viz).

pca_feature_dump.pt 의 (tidx,blk) 별 video0 full spatial feature (f,h,w,D) 를
  - 가운데 frame 선택 → (h,w,D)
  - teacher+student 토큰 합쳐 joint PCA fit (같은 색공간)
  - top-3 PC → [0,1] 정규화 → RGB 이미지
행=timestep, 열=block별 (teacher|student) 쌍.
"""
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DUMP = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/pca_feature_dump.pt"
OUT  = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/pca_spatial_teacher_student_matchS2.png"

d = torch.load(DUMP, map_location='cpu')
keys = [eval(k) for k in d.keys()]
tidxs = sorted(set(k[0] for k in keys))
blks  = sorted(set(k[1] for k in keys))
print(f"tidx={tidxs} blk={blks}")

def chan_pca_rgb(T_hwD, S_hwD):
    """teacher/student (h,w,D) → joint PCA top3 → RGB (h,w,3) 각각."""
    ht, wt, D = T_hwD.shape
    hs, ws, _ = S_hwD.shape
    Tt = T_hwD.reshape(-1, D).double()
    Ss = S_hwD.reshape(-1, D).double()
    X = torch.cat([Tt, Ss], 0)
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    U, Sg, Vh = torch.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vh[:3].T                       # (N,3)
    # [0,1] 정규화 (1~99 percentile clip)
    lo = torch.quantile(proj, 0.01, dim=0); hi = torch.quantile(proj, 0.99, dim=0)
    proj = ((proj - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    nt = Tt.shape[0]
    Trgb = proj[:nt].reshape(ht, wt, 3).numpy()
    Srgb = proj[nt:].reshape(hs, ws, 3).numpy()
    return Trgb, Srgb

nr = len(tidxs); nc = len(blks) * 2
fig, axes = plt.subplots(nr, nc, figsize=(3.2*nc, 3.4*nr), squeeze=False)
for i, tx in enumerate(tidxs):
    for j, bl in enumerate(blks):
        key = str((tx, bl))
        axT = axes[i][2*j]; axS = axes[i][2*j+1]
        if key not in d:
            axT.axis('off'); axS.axis('off'); continue
        e = d[key]
        T = e['teacher']; S = e['student']      # (f,h,w,D)
        fT = T.shape[0]//2; fS = S.shape[0]//2   # 가운데 frame
        Trgb, Srgb = chan_pca_rgb(T[fT], S[fS])
        clean = 'CLEAN' if tx==max(tidxs) else ('NOISE' if tx==min(tidxs) else 'MID')
        axT.imshow(Trgb); axT.set_title(f"teacher b{bl}\ntidx{tx} t={e['t']:.0f} {clean}", fontsize=9); axT.axis('off')
        axS.imshow(Srgb); axS.set_title(f"student b{bl}", fontsize=9); axS.axis('off')
fig.suptitle('Channel-PCA → RGB spatial feature map (teacher vs student, joint PCA per cell)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT, dpi=85, bbox_inches='tight')
print("saved:", OUT)
