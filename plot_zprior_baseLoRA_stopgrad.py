import json, glob, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
import wandb

def parse_local(run_glob, key='train/diff_loss_z_prior'):
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
    out.sort()
    return [s for s,_ in out],[v for _,v in out]

def api_local(rid, key='train/diff_loss_z_prior'):
    r=wandb.Api().run(f"kplove0503/kinemadae/{rid}")
    rows=sorted([(x['_step'],x[key]) for x in r.scan_history(keys=['_step',key]) if x.get(key) is not None])
    return [s for s,_ in rows],[v for _,v in rows]

def ema(v,a=0.1):
    out=[]; c=v[0] if v else 0
    for x in v: c=a*x+(1-a)*c; out.append(c)
    return out

BASE="/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-adaptive-Bfix"
# v2 = 현재 도는 baseLoRA+stopgrad (로컬 .wandb 파싱)
s_v2, zp_v2 = parse_local(f"{BASE}/wandb/run-*nm6bcyff*/run-nm6bcyff.wandb")
# 이전 24z63sm1 (kill된 baseLoRA+stopgrad) 도 로컬 파싱해서 이어붙임
try:
    s_v1, zp_v1 = parse_local(f"{BASE}/wandb/run-*24z63sm1*/run-24z63sm1.wandb")
except Exception: s_v1, zp_v1 = [], []
# v43 (충돌, no init) + s2bx2mpq (충돌, baseLoRA) — finished, API 됨
s_v43, zp_v43 = api_local("r8jp2j42")
s_s2,  zp_s2  = api_local("s2bx2mpq")

print(f"v2(local) n={len(zp_v2)}, 24z63sm1(local) n={len(zp_v1)}, v43 n={len(zp_v43)}, s2bx2mpq n={len(zp_s2)}")
if zp_v2: print(f"  v2 z_prior: 처음={[round(x,3) for x in zp_v2[:5]]} 최근={[round(x,3) for x in zp_v2[-5:]]}")

fig,ax=plt.subplots(figsize=(10,5.5))
series=[
    (s_v43, zp_v43, "v43 (conflict, pretrained-Wan init)", "tab:red"),
    (s_s2,  zp_s2,  "baseLoRA (conflict)", "tab:orange"),
    (s_v1,  zp_v1,  "baseLoRA+stopgrad (prev, killed)", "tab:cyan"),
    (s_v2,  zp_v2,  "baseLoRA+stopgrad (RUNNING)", "tab:blue"),
]
for s,zp,lbl,c in series:
    if not zp: continue
    ax.plot(s,zp,c=c,alpha=0.13,lw=0.7)
    ax.plot(s,ema(zp),c=c,lw=2.0,label=lbl)
ax.axhline(0.05,ls='--',c='gray',lw=1.2,label='baseline pure Wan (0.05)')
ax.set_yscale('log'); ax.set_xlabel('optimizer step'); ax.set_ylabel('z_prior diffusion loss (log)')
ax.set_title('z_prior: align conflict (spike) vs stop_grad removal + baseLoRA init', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.3)
plt.tight_layout()
out=f"{BASE}/zprior_baseLoRA_stopgrad_compare.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
print("saved:", out)
