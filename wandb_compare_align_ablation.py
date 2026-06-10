"""
diffusion z_prior / z_main loss 비교 plot (align 충돌 검증):
  r8jp2j42  v43      : align ON  + diffusion (z_prior spike)
  khtf409b  ablation : align OFF + diffusion (diffusion_only, align_loss×0)
  kdkr33g6  baseline : 순수 Wan I2V @256 (단일 diffusion loss, z 분리 없음) — reference

스타일: 기존 wandb_compare_4runs_train.py 동일 (EMA 0.99 + raw 0.20 alpha, log scale).
"""
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

api = wandb.Api()

# (run path, color, {plot_metric: actual_wandb_key})
RUNS = {
    "E2E Training w alignment loss": (
        "kplove0503/kinemadae/r8jp2j42", "tab:red",
        {"z_prior": "train/diff_loss_z_prior", "z_main": "train/diff_loss_z_main"},
    ),
    "E2E Training w/o alignment loss": (
        "kplove0503/kinemadae/khtf409b", "tab:blue",
        {"z_prior": "train/diff_loss_z_prior", "z_main": "train/diff_loss_z_main"},
    ),
    "baseline (pure Wan @256)": (
        "kplove0503/kinemadae-dit/kdkr33g6", "gray",
        # baseline 은 base/residual 분리 없는 단일 latent → base latent 패널에만 reference.
        # residual latent 은 우리 구조 전용(zero-init head)이라 baseline 대응 없음.
        {"z_prior": "train/loss"},
    ),
}

PANELS = [
    ("z_prior", "Base Latent diffusion loss (↓)"),
    ("z_main",  "Residual Latent diffusion loss (↓)"),
]


def ema_smooth(values, decay=0.99):
    if len(values) == 0:
        return []
    out, cur = [], float(values[0])
    for v in values:
        cur = decay * cur + (1.0 - decay) * float(v)
        out.append(cur)
    return out


# fetch
data = {}
for label, (path, color, mmap) in RUNS.items():
    print(f"loading {path} ...")
    r = api.run(path)
    df = r.history(pandas=True, samples=20000).sort_values("_step")
    data[label] = df
    print(f"  rows={len(df)}, step=[{df['_step'].min()}, {df['_step'].max()}]")


def plot_panel(ax, panel_key, title, xlim):
    for label, (path, color, mmap) in RUNS.items():
        key = mmap.get(panel_key)
        df = data[label]
        if key not in df.columns:
            continue
        sub = df[["_step", key]].dropna()
        sub = sub[sub[key].apply(lambda v: isinstance(v, (int, float)))].sort_values("_step")
        if len(sub) == 0:
            continue
        steps = sub["_step"].values
        raw = sub[key].astype(float).values
        is_ref = (label.startswith("baseline"))
        ls = "--" if is_ref else "-"
        ax.plot(steps, raw, color=color, linewidth=0.8, alpha=0.15)
        ax.plot(steps, ema_smooth(raw), label=label, color=color,
                linewidth=2.0, linestyle=ls, alpha=1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(fontsize=8, loc="best")


def make_plot(xlim, out_path, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (k, t) in zip(axes, PANELS):
        plot_panel(ax, k, t, xlim)
    fig.suptitle(f"Diffusion loss: align ON vs OFF vs baseline{suffix}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


OUT = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix"
make_plot(None,        f"{OUT}/wandb_compare_align_ablation_full.png",   " (full)")
make_plot((0, 1050),   f"{OUT}/wandb_compare_align_ablation_0-1050.png", " (step 0-1050, ablation range)")
