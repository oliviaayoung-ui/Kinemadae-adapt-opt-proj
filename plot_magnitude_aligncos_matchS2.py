"""matchS2 측정 [ALL_TS] 로그 → per-timestep lines over blocks (3-panel) + heatmap.

panel1 magnitude ratio(native snorm/tnorm), panel2 align cosine(teacher↔student),
panel3 eps cosine(noise robustness). rank 평균 = 16-video 집계.
"""
import re, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm; from matplotlib.colors import Normalize

LOG = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/measure_matchS2_8gpu.log"
OUT_LINES = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/magnitude_aligncos_per_timestep_lines_matchS2.png"
OUT_HEAT  = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix/magnitude_aligncos_per_ts_block_matchS2.png"

pat = re.compile(r"rank=(\d+) tidx=(\d+) t=([\d.]+) block(\d+) tnorm=([\d.]+) snorm=([\d.]+) snorm_i=([\d.]+) ratio=([\d.]+) align_cos=([-\d.]+) eps_cos=([-\d.]+)")
rows = []
for ln in open(LOG):
    m = pat.search(ln)
    if m:
        g = m.groups()
        rows.append((int(g[0]), int(g[1]), float(g[2]), int(g[3]), float(g[4]), float(g[5]), float(g[6]), float(g[7]), float(g[8]), float(g[9])))
print("rows:", len(rows))
NT = max(r[1] for r in rows) + 1; NB = max(r[3] for r in rows) + 1
ratio = np.full((NT, NB), np.nan); acos = np.full((NT, NB), np.nan); ecos = np.full((NT, NB), np.nan)
acc = {}; ts = np.zeros(NT)
for rk, tix, t, bl, tn, sn, sni, ra, ac, ec in rows:
    acc.setdefault((tix, bl), []).append((ra, ac, ec)); ts[tix] = t
for (tix, bl), vals in acc.items():
    v = np.array(vals); ratio[tix, bl] = v[:, 0].mean(); acos[tix, bl] = v[:, 1].mean(); ecos[tix, bl] = v[:, 2].mean()
lb = NB - 1
print(f"집계 rank/(tix,bl) = {len(acc[(0,0)])}")
print(f"clean+last(b{lb}): ratio={ratio[-1,lb]:.3f} align_cos={acos[-1,lb]:.3f} eps_cos={ecos[-1,lb]:.3f}")
print(f"noise+last: ratio={ratio[0,lb]:.3f} align_cos={acos[0,lb]:.3f} eps_cos={ecos[0,lb]:.3f}")
print(f"overall: ratio={np.nanmean(ratio):.3f} acos={np.nanmean(acos):.3f} ecos={np.nanmean(ecos):.3f}")

# --- plot: per-timestep lines (3 panel) ---
cmap = cm.viridis; norm = Normalize(vmin=0, vmax=NT - 1); blocks = np.arange(NB)
fig, ax = plt.subplots(1, 3, figsize=(22, 6.5))
for tix in range(NT):
    c = cmap(norm(tix))
    ax[0].plot(blocks, ratio[tix], color=c, lw=0.8, alpha=0.7)
    ax[1].plot(blocks, acos[tix], color=c, lw=0.8, alpha=0.7)
    ax[2].plot(blocks, ecos[tix], color=c, lw=0.8, alpha=0.7)
for a, dat, lab in [(ax[0], ratio, 'ratio'), (ax[1], acos, 'align_cos'), (ax[2], ecos, 'eps_cos')]:
    a.plot(blocks, dat[NT - 1], color='red', lw=2.5, label='clean(tidx%d)' % (NT - 1))
    a.plot(blocks, dat[0], color='blue', lw=2.5, ls='--', label='noise(tidx0)')
    a.set_xlabel('block (0->deep)'); a.set_ylabel(lab); a.grid(alpha=0.3); a.legend(fontsize=8); a.axvline(NB - 1, color='gray', ls=':', alpha=0.5)
ax[0].set_title('magnitude ratio (native) per timestep', fontweight='bold')
ax[1].set_title('align cosine per timestep', fontweight='bold')
ax[2].set_title('eps cosine (noise robustness) per timestep', fontweight='bold')
sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
fig.colorbar(sm, ax=ax[2], label='tidx (0=noise->%d=clean)' % (NT - 1))
fig.suptitle('[matchS2 ckpt-4000, 16-video] Per-timestep lines over blocks (native snorm)  red=clean blue=noise', fontsize=13)
fig.tight_layout(); fig.savefig(OUT_LINES, dpi=100, bbox_inches='tight'); print("saved:", OUT_LINES)

# --- plot: heatmap (ratio + align_cos + eps_cos) ---
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for a, dat, title, cmp, vr in [(ax[0], ratio, 'magnitude ratio', 'viridis', (None, None)),
                                 (ax[1], acos, 'align cosine', 'magma', (0.4, 1.0)),
                                 (ax[2], ecos, 'eps cosine (noise robust)', 'cividis', (0.3, 1.0))]:
    im = a.imshow(dat, aspect='auto', cmap=cmp, origin='lower', vmin=vr[0], vmax=vr[1])
    a.set_title(title + ' [matchS2]', fontweight='bold'); a.set_xlabel('block'); a.set_ylabel('tidx (0=noise->clean)'); plt.colorbar(im, ax=a)
fig.tight_layout(); fig.savefig(OUT_HEAT, dpi=100, bbox_inches='tight'); print("saved:", OUT_HEAT)
