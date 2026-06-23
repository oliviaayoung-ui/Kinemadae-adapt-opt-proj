"""PCA 시각화: 선택한 (timestep, block) 의 teacher vs student feature 분포.

pca_feature_dump.pt 의 (tidx,blk) 별 teacher/student 토큰을 joint PCA → 2D scatter.
  - teacher+student 합쳐서 PC 축 계산 (같은 공간에 투영) → 두 분포 위치/겹침 비교.
"""
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DUMP = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/pca_feature_dump.pt"
OUT  = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/pca_teacher_student_grid.png"

d = torch.load(DUMP, map_location='cpu')
keys = [eval(k) for k in d.keys()]          # (tidx, blk) 튜플
tidxs = sorted(set(k[0] for k in keys))
blks  = sorted(set(k[1] for k in keys))
print(f"tidx={tidxs} blk={blks}  총 {len(keys)} 패널")

def joint_pca(T, S):
    X = torch.cat([T, S], dim=0).double()
    Xc = X - X.mean(0, keepdim=True)
    # SVD 로 top-2 PC
    U, Sg, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ Vh[:2].T                       # (N, 2)
    var = (Sg[:2]**2 / (Sg**2).sum()).tolist()
    nt = T.shape[0]
    return pc[:nt].numpy(), pc[nt:].numpy(), var

nr, nc = len(tidxs), len(blks)
fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr), squeeze=False)
for i, tx in enumerate(tidxs):
    for j, bl in enumerate(blks):
        ax = axes[i][j]
        key = str((tx, bl))
        if key not in d:
            ax.axis('off'); continue
        e = d[key]
        Tp, Sp, var = joint_pca(e['teacher'], e['student'])
        ax.scatter(Tp[:,0], Tp[:,1], s=4, alpha=0.35, c='tab:red',  label='teacher (Wan 16ch)')
        ax.scatter(Sp[:,0], Sp[:,1], s=4, alpha=0.35, c='tab:blue', label='student (geoprior 32ch)')
        tstr = f"t={e['t']:.0f}"
        clean = 'CLEAN' if tx == max(tidxs) else ('NOISE' if tx == min(tidxs) else 'MID')
        ax.set_title(f"tidx={tx} ({tstr}, {clean}) · block{bl}\nPC var={var[0]*100:.0f}%,{var[1]*100:.0f}%", fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(alpha=0.2)
        if i==0 and j==0: ax.legend(fontsize=8, markerscale=2)
fig.suptitle('Teacher vs Student feature — joint PCA per (timestep, block)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT, dpi=100, bbox_inches='tight')
print("saved:", OUT)
