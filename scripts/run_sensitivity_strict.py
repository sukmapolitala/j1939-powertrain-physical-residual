from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from continuous_state_residual import ContinuousStateResidualDetector
from run_primary_strict import (
    build_protocol, metrics, fit_baselines, score_baselines,
    CONTEXTS, RESPONSES, ALL_FEATURES, RESOLUTION
)

def proposed_run(ms=20,q=.99):
    p=build_protocol(ms)
    det=ContinuousStateResidualDetector(
        CONTEXTS,RESPONSES,degree=2,calibration_quantile=q,
        ewma_alpha=.25,persistence_samples=3,residual_z_clip=8
    ).fit(p["train"],p["calib"])
    pr=det.predict(p["test"])
    m=metrics(p["y"],pr["ewma_score"],pr["alarm"])
    ep=det.predict(p["event"])
    idx=np.flatnonzero(ep["alarm"])
    m["detection_delay_s"]=float(p["event"].timestamp.iloc[idx[0]]-p["onset"]) if len(idx) else np.nan
    return p,det,pr,m

def resolution_noise(df,factor,rng):
    out=df.copy()
    for f,res in RESOLUTION.items():
        x=pd.to_numeric(out[f],errors="coerce").to_numpy(float)
        noise=rng.normal(0.0,factor*res,size=len(x))
        out[f]=x+noise
    return out

def response_packet_loss(df,p,rng):
    out=df.copy()
    for f in RESPONSES:
        mask=rng.random(len(out))<p
        out.loc[mask,f]=np.nan
        vc=f"valid__{f}"
        if vc in out: out.loc[mask,vc]=0
    return out

def full_response_dropout(df,f):
    out=df.copy(); out[f]=np.nan
    vc=f"valid__{f}"
    if vc in out: out[vc]=0
    return out

if __name__=="__main__":
    results=ROOT/"results"

    # Grid sensitivity
    grid=[]
    for ms in [20,50,100]:
        p,det,pr,m=proposed_run(ms,.99)
        grid.append({"grid_ms":ms,**m,
                     "train_rows":len(p["train"]),"calibration_rows":len(p["calib"]),
                     "heldout_healthy_rows":len(p["healthy_test"]),"event_rows":len(p["event"])})
    pd.DataFrame(grid).to_csv(results/"time_grid_sensitivity.csv",index=False)

    # Threshold sensitivity at locked 20ms grid.
    thr=[]
    for q in [.975,.99,.995]:
        p,det,pr,m=proposed_run(20,q)
        thr.append({"calibration_quantile":q,**m,
                    "raw_threshold":det.raw_threshold,"ewma_threshold":det.ewma_threshold})
    pd.DataFrame(thr).to_csv(results/"threshold_sensitivity.csv",index=False)

    # Robustness uses the LOCKED primary 20ms model.
    p,det,pr,m=proposed_run(20,.99)
    rng=np.random.default_rng(20260828)
    robust=[]

    for factor in [.5,1,2,5]:
        stressed=resolution_noise(p["test"],factor,rng)
        pp=det.predict(stressed)
        mm=metrics(p["y"],pp["ewma_score"],pp["alarm"])
        robust.append({"stress":"SPN_resolution_noise","level":factor,**mm})

    for loss in [.05,.10,.20]:
        stressed=response_packet_loss(p["test"],loss,rng)
        pp=det.predict(stressed)
        mm=metrics(p["y"],pp["ewma_score"],pp["alarm"])
        robust.append({"stress":"response_packet_loss","level":loss,**mm})

    for f in RESPONSES:
        stressed=full_response_dropout(p["test"],f)
        pp=det.predict(stressed)
        mm=metrics(p["y"],pp["ewma_score"],pp["alarm"])
        robust.append({"stress":"complete_response_dropout","level":f,**mm})

    pd.DataFrame(robust).to_csv(results/"robustness_results.csv",index=False)

    print("TIME GRID SENSITIVITY")
    print(pd.DataFrame(grid).to_string(index=False))
    print("\nTHRESHOLD SENSITIVITY")
    print(pd.DataFrame(thr).to_string(index=False))
    print("\nROBUSTNESS")
    print(pd.DataFrame(robust).to_string(index=False))
