"""noise robustness PCA viz (= 3번째 panel eps-invariance 의 spatial 버전).

pca_feature_dump.pt 의 (tidx,blk) 별 teacher_eps (= 같은 z0+t, 다른 noise N개 의 teacher feature)를
  - 가운데 frame 선택 → (h,w,D)
  - N개 eps 를 합쳐 joint PCA fit (= 같은 색공간) → top-3 PC → RGB
  - 한 행 = (tidx, blk), 열 = eps0..eps_{N-1}
robust 면 한 행의 N 열이 (거의) 동일, noise 에 민감하면 열마다 다름.
열 위에 옆 eps 와의 cosine 도 표기.
"""
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch.nn.functional as F

DUMP = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/pca_feature_dump.pt"
OUT  = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/pca_noise_robust_matchS2.png"

d = torch.load(DUMP, map_location='cpu')
keys = sorted([eval(k) for k in d.keys()])
# teacher_eps 있는 키만
keys = [k for k in keys if 'teacher_eps' in d[str(k)]]
if not keys:
    raise SystemExit("teacher_eps 없음 — measure_pca_dump 를 수정된 코드로 재실행 필요")
N_eps = len(d[str(keys[0])]['teacher_eps'])
tidxs = sorted(set(k[0] for k in keys)); blks = sorted(set(k[1] for k in keys))
print(f"keys={keys} N_eps={N_eps}")

def joint_pca_rgb(feats_hwD):
    """[N] × (h,w,D) → joint PCA top3 → [N] × (h,w,3). 같은 색공간."""
    h, w, D = feats_hwD[0].shape
    X = torch.cat([f.reshape(-1, D).double() for f in feats_hwD], 0)
    mu = X.mean(0, keepdim=True); Xc = X - mu
    U, Sg, Vh = torch.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vh[:3].T
    lo = torch.quantile(proj, 0.01, dim=0); hi = torch.quantile(proj, 0.99, dim=0)
    proj = ((proj - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    n = h * w
    return [proj[i*n:(i+1)*n].reshape(h, w, 3).numpy() for i in range(len(feats_hwD))]

nr = len(keys); nc = N_eps
fig, axes = plt.subplots(nr, nc, figsize=(3.3*nc, 3.5*nr), squeeze=False)
for i, (tx, bl) in enumerate(keys):
    e = d[str((tx, bl))]
    eps_feats = e['teacher_eps']                      # [N] × (f,h,w,D)
    fmid = eps_feats[0].shape[0] // 2
    mids = [ef[fmid] for ef in eps_feats]             # [N] × (h,w,D)
    rgbs = joint_pca_rgb(mids)
    # 이웃 eps cosine — align loss 와 동일한 per-token (dim=-1 정규화 후 토큰 평균)
    toks = [F.normalize(m.reshape(-1, m.shape[-1]).double(), dim=-1) for m in mids]
    clean = 'CLEAN' if tx == max(tidxs) else ('NOISE' if tx == min(tidxs) else 'MID')
    for j in range(N_eps):
        ax = axes[i][j]
        ax.imshow(rgbs[j]); ax.axis('off')
        if j == 0:
            ax.set_title(f"b{bl} tidx{tx} t={e['t']:.0f} {clean}\neps0", fontsize=9)
        else:
            cos = (toks[0] * toks[j]).sum(-1).mean().item()
            ax.set_title(f"eps{j}  cos(0,{j})={cos:.3f}", fontsize=9)
fig.suptitle(f'[matchS2] Noise robustness PCA: same z0+t, {N_eps} different noise (joint PCA→RGB per row)\n'
             f'한 행 N열이 동일 = noise-invariant', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT, dpi=85, bbox_inches='tight')
print("saved:", OUT)
