#!/usr/bin/env python
# 200 step 마다 align_loss / rec_loss 를 run 폴더 아래에 plot (local .wandb 직접 파싱 → wandb UI delay 무관)
import time, glob, os, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

BASE = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix"
RUN_GLOB = BASE + "/wandb/run-*fj63rvum*/run-*.wandb"
OUT = BASE + "/results/kinemadae_stage1_bn_lora_align40_bs4_b_proj_teacherfrozen-lr8.00e-05-bs4-rs256-sr2-fr17"
os.makedirs(OUT, exist_ok=True)
INTERVAL_STEPS = 200      # 이 step 수만큼 늘 때마다 plot 갱신
POLL_SEC = 30


def parse():
    fs = glob.glob(RUN_GLOB)
    if not fs:
        return []
    ds = DataStore(); ds.open_for_scan(fs[0]); rows = []
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        try:
            r.ParseFromString(d)
        except Exception:
            continue
        if r.WhichOneof("record_type") == "history":
            dd = {}
            for it in r.history.item:
                k = it.key if it.key else ".".join(it.nested_key)
                try:
                    dd[k] = json.loads(it.value_json)
                except Exception:
                    pass
            rows.append(dd)
    return rows


def series(rows, key):
    return [(r.get('_step', i), r[key]) for i, r in enumerate(rows) if key in r]


def make_plot(rows):
    al = series(rows, 'train/align_loss')
    rc = series(rows, 'train/rec_loss')
    cur = rows[-1].get('_step', len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    if al:
        xs, ys = zip(*al)
        ax[0].plot(xs, ys, lw=1.2, color='tab:blue')
        ax[0].set_title(f'align_loss  (last={ys[-1]:.4f})')
        ax[0].set_xlabel('step'); ax[0].set_ylabel('align_loss'); ax[0].grid(alpha=0.3)
    if rc:
        xs, ys = zip(*rc)
        ax[1].plot(xs, ys, lw=1.2, color='tab:orange')
        ax[1].set_title(f'rec_loss  (last={ys[-1]:.1f})')
        ax[1].set_xlabel('step'); ax[1].set_ylabel('rec_loss'); ax[1].grid(alpha=0.3)
    fig.suptitle(f'teacherfrozen (stage1+BN+LoRA, align only)  step={cur}  n={len(rows)}')
    fig.savefig(OUT + '/align_rec_loss.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return cur


last_n = -INTERVAL_STEPS
while True:
    rows = parse()
    n = len(rows)
    if n > 0 and (n - last_n) >= INTERVAL_STEPS:
        try:
            cur = make_plot(rows)
            print(f"plotted: step={cur}  n={n}  -> {OUT}/align_rec_loss.png", flush=True)
            last_n = n
        except Exception as e:
            print(f"plot error: {e}", flush=True)
    time.sleep(POLL_SEC)
