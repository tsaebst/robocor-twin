import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def series(rows, key, default=np.nan):
    out=[]
    for r in rows:
        v=r
        ok=True
        for k in key:
            if isinstance(v,dict) and k in v:
                v=v[k]
            else:
                ok=False
                break
        if (not ok) or v is None:
            out.append(default)
        else:
            try: out.append(float(v))
            except Exception: out.append(default)
    return np.asarray(out,dtype=float)

def rolling_mean(x,w):
    x=np.asarray(x,dtype=float)
    w=int(w)
    if w<=1 or x.size==0: return x
    if x.size<w:
        k=np.ones(max(1,x.size))/max(1,x.size)
        return np.convolve(x,k,mode="same")
    k=np.ones(w)/w
    return np.convolve(x,k,mode="same")

def ffill(x, fill0=0.0):
    x=np.asarray(x,dtype=float)
    if x.size==0: return x
    if not np.isfinite(x[0]): x[0]=fill0
    for i in range(1,len(x)):
        if not np.isfinite(x[i]): x[i]=x[i-1]
    return x

def compute_query_rate(t,k):
    t=np.asarray(t,dtype=float)
    k=np.asarray(k,dtype=float)
    if t.size==0: return t
    dt=np.diff(t,prepend=t[0])
    dt=np.maximum(dt,1.0)
    dq=np.diff(k,prepend=k[0])
    return dq/dt

def load_ppo(path):
    rows=read_jsonl(path)
    upd=[r for r in rows if r.get("event") in ("oracle_update","oracle_update_skipped")]
    if not upd:
        raise SystemExit(f"[PPO] no oracle_update(_skipped) in {path}")
    t=ffill(series(upd,["t_global"]),0.0)
    k=ffill(series(upd,["k_used_total"]),0.0)
    th=ffill(series(upd,["acc","acc/theta_mean_abs"]),0.0)
    run_id=upd[0].get("run_id","ppo")
    return {"name":f"PPO (run={run_id})","t":t,"k":k,"theta_err":th}

def load_baseline_as_is(path,label):
    rows=read_jsonl(path)

    # Prefer step events
    steps=[r for r in rows if r.get("event")=="step"]
    if steps:
        # time axis
        if any("t_global" in r for r in steps):
            t=series(steps,["t_global"])
            # fill NaNs by simple forward-fill + +1 fallback
            t=ffill(t,0.0)
            for i in range(1,len(t)):
                if t[i] < t[i-1]:
                    t[i]=t[i-1]+1
        else:
            t=np.arange(len(steps),dtype=float)

        # budget axis
        if any("k_used_total" in r for r in steps):
            k=ffill(series(steps,["k_used_total"]),0.0)
        else:
            # fallback from queried_oracle
            q=series(steps,["queried_oracle"],default=0.0)
            q=np.nan_to_num(q,nan=0.0)
            q=(q>0.5).astype(float)
            k=np.cumsum(q)

        # theta error: either per-step logged, or step-aligned from calib_update
        th_step=series(steps,["acc","acc/theta_mean_abs"])
        if np.any(np.isfinite(th_step)):
            theta_err=ffill(th_step,0.0)
        else:
            # align from calib_update events if present
            upd=[r for r in rows if r.get("event")=="calib_update"]
            ut=series(upd,["t_global"]) if upd else np.asarray([],dtype=float)
            uth=series(upd,["acc","acc/theta_mean_abs"]) if upd else np.asarray([],dtype=float)
            theta_err=np.full_like(t,np.nan,dtype=float)
            last=np.nan; j=0
            if ut.size>0 and any("t_global" in r for r in steps):
                for i in range(len(t)):
                    while j<len(ut) and ut[j] <= t[i]:
                        if np.isfinite(uth[j]): last=uth[j]
                        j+=1
                    theta_err[i]=last
            else:
                # no comparable time; just forward-fill in update order across steps
                if uth.size>0:
                    ut_idx=np.round(np.linspace(0,len(t)-1,num=len(uth))).astype(int)
                    last=np.nan; j=0
                    for i in range(len(t)):
                        while j<len(ut_idx) and ut_idx[j] <= i:
                            if np.isfinite(uth[j]): last=uth[j]
                            j+=1
                        theta_err[i]=last
            theta_err=ffill(theta_err,0.0)

        # name
        start=next((r for r in rows if r.get("event")=="start"),{})
        run_id=start.get("run_id",label)
        return {"name":f"{label} (run={run_id})","t":t,"k":k,"theta_err":theta_err}

    upd=[r for r in rows if r.get("event") in ("oracle_update","oracle_update_skipped")]
    if upd:
        t=ffill(series(upd,["t_global"]),0.0)
        k=ffill(series(upd,["k_used_total"]),0.0)
        th=ffill(series(upd,["acc","acc/theta_mean_abs"]),0.0)
        run_id=upd[0].get("run_id",label)
        return {"name":f"{label} (run={run_id})","t":t,"k":k,"theta_err":th}

    raise SystemExit(f"[{label}] can't find usable events in {path} (no step, no oracle_update)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ppo_log", required=True)
    ap.add_argument("--baseline", action="append", default=[], help='e.g. "A1=logs/a1.jsonl"')
    ap.add_argument("--out", default="plots/overlay_as_is.png")
    ap.add_argument("--roll", type=int, default=51)
    args=ap.parse_args()

    ppo=load_ppo(args.ppo_log)
    baselines=[]
    for spec in args.baseline:
        if "=" not in spec:
            raise SystemExit(f"Bad --baseline '{spec}', expected Label=path")
        label,path=spec.split("=",1)
        baselines.append(load_baseline_as_is(path.strip(), label.strip()))

    # compute query-rate
    ppo_qr=rolling_mean(compute_query_rate(ppo["t"], ppo["k"]), args.roll)
    ppo_th=rolling_mean(ppo["theta_err"], args.roll)

    base_qr=[]
    base_th=[]
    for b in baselines:
        base_qr.append(rolling_mean(compute_query_rate(b["t"], b["k"]), args.roll))
        base_th.append(rolling_mean(b["theta_err"], args.roll))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(13.5,5.5))
    ax.plot(ppo["t"], ppo_qr, linewidth=2.0, label=f"{ppo['name']} query rate")

    for b,qr in zip(baselines, base_qr):
        ax.plot(b["t"], qr, linewidth=2.0, linestyle="--", label=f"{b['name']} query rate")

    ax.set_xlabel("Time axis (t_global if present, else step index)")
    ax.set_ylabel("Query rate (Δk / Δt)")
    ax.set_title("Oracle query adaptation vs calibration accuracy (as-is logs)")

    ax2=ax.twinx()
    ax2.plot(ppo["t"], ppo_th, linewidth=2.0, linestyle=":", label=f"{ppo['name']} theta error")

    for b,th in zip(baselines, base_th):
        ax2.plot(b["t"], th, linewidth=2.0, linestyle="-.", label=f"{b['name']} theta error")

    ax2.set_ylabel("Theta error (mean abs; lower is better)")

    h1,l1=ax.get_legend_handles_labels()
    h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print("Saved:", args.out)

if __name__=="__main__":
    main()
