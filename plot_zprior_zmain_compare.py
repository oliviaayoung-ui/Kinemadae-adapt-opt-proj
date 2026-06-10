import json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
import wandb

def parse_local(run_glob, key):
    path=glob.glob(run_glob)[0]
    ds=DataStore(); ds.open_for_scan(path)
    out=[]
    while True:
        try: data=ds.scan_data()
        except Exception: break
        if data is None: break
        rec=pb.Record()
        try: rec.ParseFromString(data)
        except Exception: continue
        if rec.WhichOneof("record_type")=="history":
            d={}
            for it in rec.history.item:
                k=it.key if it.key else ".".join(it.nested_key)
                try: d[k]=json.loads(it.value_json)
                except Exception: pass
            if key in d and '_step' in d: out.append((d['_step'], d[key]))
    out.sort(); return [s for s,_ in out],[v for _,v in out]

def api_local(rid, key):
    r=wandb.Api().run(f"kplove0503/kinemadae/{rid}")
    rows=sorted([(x['_step'],x[key]) for x in r.scan_history(keys=['_step',key]) if x.get(key) is not None])
    return [s for s,_ in rows],[v for _,v in rows]

def ema(v,a=0.1):
    out=[]; c=v[0] if v else 0
    for x in v: c=a*x+(1-a)*c; out.append(c)
    return out

BASE="/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix"
RUNS=[
    ("api", "r8jp2j42", "conflict + pretrained init (v43)", "tab:red"),
    ("api", "s2bx2mpq", "conflict + baseLoRA init", "tab:orange"),
    ("api", "s1ytwor7", "stop_grad + pretrained init", "tab:green"),
    ("local", f"{BASE}/wandb/run-*nm6bcyff*/run-nm6bcyff.wandb", "stop_grad + baseLoRA init (RUNNING)", "tab:blue"),
]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, key, title in [(axes[0],'train/diff_loss_z_prior','z_prior (base latent)'),
                       (axes[1],'train/diff_loss_z_main','z_main (residual latent)')]:
    for kind, ref, lbl, c in RUNS:
        try:
            s, v = (parse_local(ref, key) if kind=="local" else api_local(ref, key))
        except Exception: s, v = [], []
        if not v: continue
        ax.plot(s, v, c=c, alpha=0.12, lw=0.7)
        ax.plot(s, ema(v), c=c, lw=2.0, label=lbl)
    if 'z_prior' in title: ax.axhline(0.05, ls='--', c='gray', lw=1.2, label='baseline (0.05)')
    ax.set_yscale('log'); ax.set_xlabel('optimizer step')
    ax.set_ylabel('diffusion loss (log)'); ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3)
fig.suptitle('2x2 ablation: align conflict(O/X) x init(pretrained/baseLoRA)', fontsize=13, fontweight='bold')
plt.tight_layout()
out=f"{BASE}/zprior_zmain_2x2.png"
plt.savefig(out, dpi=120, bbox_inches='tight'); print("saved:", out)
# z_main 추세 출력
for kind, ref, lbl, c in RUNS:
    try: s,v=(parse_local(ref,'train/diff_loss_z_main') if kind=="local" else api_local(ref,'train/diff_loss_z_main'))
    except: v=[]
    if v: print(f"  {lbl}: z_main 처음={round(v[0],3)} 최근={round(v[-1],3)} (n={len(v)})")
