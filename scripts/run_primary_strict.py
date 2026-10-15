from __future__ import annotations
from pathlib import Path
import sys, json, math, hashlib, joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from continuous_state_residual import ContinuousStateResidualDetector

CONTEXTS=["accelerator_pedal_pct","driver_demand_torque_pct","vehicle_speed_kph"]
RESPONSES=["engine_demand_torque_pct","actual_engine_torque_pct","engine_load_pct",
           "engine_speed_rpm","fuel_rate_lph","intake_manifold_pressure_kpa"]
ALL_FEATURES=CONTEXTS+RESPONSES

# J1939 physical resolutions for stress testing (engineering units per LSB).
RESOLUTION={
    "accelerator_pedal_pct":0.4,
    "driver_demand_torque_pct":1.0,
    "vehicle_speed_kph":1/256.0,
    "engine_demand_torque_pct":1.0,
    "actual_engine_torque_pct":1.0,
    "engine_load_pct":1.0,
    "engine_speed_rpm":0.125,
    "fuel_rate_lph":0.05,
    "intake_manifold_pressure_kpa":2.0,
}

def source_dir(ms):
    return ROOT/'data'/'local_derived'/f'{ms}ms'

def load_data(ms):
    d=source_dir(ms); suffix=f"{ms}ms"
    main=pd.read_csv(d/f"labeled_TSC1__physical_features_{suffix}.csv")
    gt=pd.read_csv(d/f"labeled_TSC1__ground_truth_{suffix}.csv")
    normal=pd.read_csv(d/f"normal__physical_features_{suffix}.csv")
    y=gt["perturbation_envelope"].fillna(0).astype(int)
    onset=float(gt.loc[y==1,"timestamp"].min())
    end=float(gt.loc[y==1,"timestamp"].max())
    return main,gt,normal,onset,end

def split_forward(seg):
    n=len(seg); i1=int(.60*n); i2=int(.80*n)
    return seg.iloc[:i1].copy(),seg.iloc[i1:i2].copy(),seg.iloc[i2:].copy()

def build_protocol(ms,guard_s=.5):
    main,gt,normal,onset,end=load_data(ms)
    pre=main[main.timestamp<onset].copy()
    # Post-event data are never used for training, calibration, or primary healthy evaluation.
    event=main[(main.timestamp>=onset)&(main.timestamp<=end)].copy()

    train=[]; calib=[]; htest=[]
    for name,seg in [("normal_run",normal),("pre_event",pre)]:
        tr,ca,te=split_forward(seg)
        tr["healthy_segment"]=name; ca["healthy_segment"]=name; te["healthy_segment"]=name
        train.append(tr);calib.append(ca);htest.append(te)
    train=pd.concat(train,ignore_index=True)
    calib=pd.concat(calib,ignore_index=True)
    healthy_test=pd.concat(htest,ignore_index=True)

    # models see physical fields only; segment is metadata used only for stratified reporting
    train_x=train.drop(columns="healthy_segment")
    calib_x=calib.drop(columns="healthy_segment")
    event_x=event.reset_index(drop=True)
    healthy_x=healthy_test.drop(columns="healthy_segment").reset_index(drop=True)

    test=pd.concat([healthy_x,event_x],ignore_index=True)
    y=np.r_[np.zeros(len(healthy_x),dtype=int),np.ones(len(event_x),dtype=int)]
    return {
        "main":main,"gt":gt,"normal":normal,"onset":onset,"end":end,
        "train":train_x,"calib":calib_x,"healthy_test":healthy_test,
        "event":event_x,"test":test,"y":y,
    }

def metrics(y,score,alarm):
    y=np.asarray(y,int); score=np.asarray(score,float); alarm=np.asarray(alarm,int)
    tn,fp,fn,tp=confusion_matrix(y,alarm,labels=[0,1]).ravel()
    return {
        "precision":precision_score(y,alarm,zero_division=0),
        "recall":recall_score(y,alarm,zero_division=0),
        "f1":f1_score(y,alarm,zero_division=0),
        "auroc":roc_auc_score(y,score) if len(np.unique(y))>1 else np.nan,
        "auprc":average_precision_score(y,score) if len(np.unique(y))>1 else np.nan,
        "false_alarm_rate":fp/(fp+tn) if fp+tn else np.nan,
        "miss_rate":fn/(fn+tp) if fn+tp else np.nan,
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }

class UnconditionedResidual:
    def __init__(self,responses,q=.99,alpha=.25,persistence=3):
        self.responses=responses; self.q=q; self.alpha=alpha; self.persistence=persistence
    def fit(self,train,cal):
        self.imp=SimpleImputer(strategy="median")
        self.scaler=RobustScaler()
        Y=self.scaler.fit_transform(self.imp.fit_transform(train[self.responses]))
        self.med=np.median(Y,axis=0)
        mad=np.median(np.abs(Y-self.med),axis=0)
        self.rscale=np.maximum(1.4826*mad,1e-5)
        Z=np.clip((Y-self.med)/self.rscale,-8,8)
        self.cov=LedoitWolf().fit(Z).covariance_+np.eye(Z.shape[1])*1e-8
        self.inv=np.linalg.pinv(self.cov)
        c=self.raw_score(cal); self.raw_thr=float(np.quantile(c,self.q))
        ew=self._ewma(c/self.raw_thr); self.ew_thr=float(np.quantile(ew,self.q))
        return self
    def raw_score(self,df):
        Y=self.scaler.transform(self.imp.transform(df[self.responses]))
        Z=np.clip((Y-self.med)/self.rscale,-8,8)
        return np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",Z,self.inv,Z))/Z.shape[1])
    def _ewma(self,s):
        o=np.empty(len(s)); o[0]=s[0]
        for i in range(1,len(s)): o[i]=self.alpha*s[i]+(1-self.alpha)*o[i-1]
        return o
    def predict(self,df):
        raw=self.raw_score(df); norm=raw/self.raw_thr; ew=self._ewma(norm)
        ex=ew>self.ew_thr; alarm=np.zeros(len(ew),int); run=0
        for i,f in enumerate(ex):
            run=run+1 if f else 0; alarm[i]=int(run>=self.persistence)
        return raw,norm,ew,alarm

def fit_baselines(train,cal,q=.99):
    imp=SimpleImputer(strategy="median"); scaler=RobustScaler()
    Xt=scaler.fit_transform(imp.fit_transform(train[ALL_FEATURES]))
    Xc=scaler.transform(imp.transform(cal[ALL_FEATURES]))
    iso=IsolationForest(n_estimators=500,random_state=42,n_jobs=-1).fit(Xt)
    svm=OneClassSVM(nu=.01,gamma="scale").fit(Xt)
    ciso=-iso.score_samples(Xc); csvm=-svm.decision_function(Xc)
    return {
        "imputer":imp,"scaler":scaler,"iso":iso,"svm":svm,
        "threshold_iso":float(np.quantile(ciso,q)),
        "threshold_svm":float(np.quantile(csvm,q))
    }

def score_baselines(base,df):
    X=base["scaler"].transform(base["imputer"].transform(df[ALL_FEATURES]))
    iso=-base["iso"].score_samples(X)
    svm=-base["svm"].decision_function(X)
    return {
        "Isolation Forest":(iso,(iso>base["threshold_iso"]).astype(int)),
        "One-Class SVM":(svm,(svm>base["threshold_svm"]).astype(int))
    }

def detection_delay(det,event,onset):
    p=det.predict(event)
    idx=np.flatnonzero(p["alarm"])
    return float(event.timestamp.iloc[idx[0]]-onset) if len(idx) else np.nan

def block_bootstrap_ci(y,score,alarm,block_size,repeats=300,seed=42):
    y=np.asarray(y,int); score=np.asarray(score,float); alarm=np.asarray(alarm,int)
    n=len(y); starts=np.arange(0,n-block_size+1)
    rng=np.random.default_rng(seed); vals={k:[] for k in ["f1","auroc","auprc","false_alarm_rate"]}
    nb=int(np.ceil(n/block_size))
    for _ in range(repeats):
        idx=np.concatenate([np.arange(s,s+block_size) for s in rng.choice(starts,size=nb,replace=True)])[:n]
        yy=y[idx]; ss=score[idx]; aa=alarm[idx]
        if len(np.unique(yy))<2: continue
        m=metrics(yy,ss,aa)
        for k in vals: vals[k].append(m[k])
    out={}
    for k,v in vals.items():
        out[k+"_ci_low"]=float(np.quantile(v,.025))
        out[k+"_ci_high"]=float(np.quantile(v,.975))
    return out

def paired_block_diff_ci(y,score_a,alarm_a,score_b,alarm_b,block_size,repeats=300,seed=91):
    y=np.asarray(y,int); rng=np.random.default_rng(seed); n=len(y)
    starts=np.arange(0,n-block_size+1); nb=int(np.ceil(n/block_size))
    diffs_f1=[]; diffs_far=[]
    for _ in range(repeats):
        idx=np.concatenate([np.arange(s,s+block_size) for s in rng.choice(starts,size=nb,replace=True)])[:n]
        yy=y[idx]
        if len(np.unique(yy))<2: continue
        ma=metrics(yy,np.asarray(score_a)[idx],np.asarray(alarm_a)[idx])
        mb=metrics(yy,np.asarray(score_b)[idx],np.asarray(alarm_b)[idx])
        diffs_f1.append(ma["f1"]-mb["f1"])
        diffs_far.append(ma["false_alarm_rate"]-mb["false_alarm_rate"])
    return {
        "delta_f1":float(np.mean(diffs_f1)),
        "delta_f1_ci_low":float(np.quantile(diffs_f1,.025)),
        "delta_f1_ci_high":float(np.quantile(diffs_f1,.975)),
        "delta_far":float(np.mean(diffs_far)),
        "delta_far_ci_low":float(np.quantile(diffs_far,.025)),
        "delta_far_ci_high":float(np.quantile(diffs_far,.975)),
    }

def save_publication_data(protocol,ms):
    d=ROOT/"results"/"local_private"
    d.mkdir(parents=True,exist_ok=True)
    # Publication feature files: no source-network metadata exists in these derived matrices.
    protocol["main"].to_csv(d/f"physical_abnormal_run_{ms}ms.csv",index=False)
    protocol["normal"].to_csv(d/f"physical_normal_run_{ms}ms.csv",index=False)
    gt=pd.DataFrame({
        "timestamp":protocol["gt"].timestamp,
        "abnormal_state":protocol["gt"].perturbation_envelope.astype(int)
    })
    gt.to_csv(d/f"physical_state_ground_truth_{ms}ms.csv",index=False)

def primary_experiment(ms=20,q=.99):
    p=build_protocol(ms)
    det=ContinuousStateResidualDetector(
        CONTEXTS,RESPONSES,degree=2,calibration_quantile=q,
        ewma_alpha=.25,persistence_samples=3,residual_z_clip=8
    ).fit(p["train"],p["calib"])
    pred=det.predict(p["test"])
    base=fit_baselines(p["train"],p["calib"],q)
    bs=score_baselines(base,p["test"])

    rows=[]
    m=metrics(p["y"],pred["ewma_score"],pred["alarm"])
    m["detection_delay_s"]=detection_delay(det,p["event"],p["onset"])
    rows.append({"method":"Proposed state-conditioned residual",**m})
    for name,(s,a) in bs.items():
        mm=metrics(p["y"],s,a); mm["detection_delay_s"]=np.nan
        # Detection delay on event only with independently fitted fixed baseline.
        es,ea=score_baselines(base,p["event"])[name]
        idx=np.flatnonzero(ea)
        mm["detection_delay_s"]=float(p["event"].timestamp.iloc[idx[0]]-p["onset"]) if len(idx) else np.nan
        rows.append({"method":name,**mm})

    # Unconditioned and linear conditioned ablations
    unc=UnconditionedResidual(RESPONSES,q).fit(p["train"],p["calib"])
    ur,un,ue,ua=unc.predict(p["test"])
    um=metrics(p["y"],ue,ua); 
    er,en,ee,ea=unc.predict(p["event"]); idx=np.flatnonzero(ea)
    um["detection_delay_s"]=float(p["event"].timestamp.iloc[idx[0]]-p["onset"]) if len(idx) else np.nan

    lin=ContinuousStateResidualDetector(
        CONTEXTS,RESPONSES,degree=1,calibration_quantile=q,
        ewma_alpha=.25,persistence_samples=3,residual_z_clip=8
    ).fit(p["train"],p["calib"])
    lp=lin.predict(p["test"])
    lm=metrics(p["y"],lp["ewma_score"],lp["alarm"]); lm["detection_delay_s"]=detection_delay(lin,p["event"],p["onset"])

    ablation=pd.DataFrame([
        {"variant":"Unconditioned residual + EWMA",**um},
        {"variant":"Linear state-conditioned residual + EWMA",**lm},
        {"variant":"Quadratic state-conditioned residual + EWMA",**m},
    ])

    # Per-healthy-segment FAR, state model only
    seg=[]
    for name,z in p["healthy_test"].groupby("healthy_segment",sort=False):
        qz=z.drop(columns="healthy_segment")
        pp=det.predict(qz)
        seg.append({"healthy_segment":name,"n":len(qz),"false_alarm_rate":float(np.mean(pp["alarm"]))})
    seg=pd.DataFrame(seg)

    # Predictions
    pred_df=pd.DataFrame({
        "ground_truth":p["y"],
        "score_proposed":pred["ewma_score"],
        "alarm_proposed":pred["alarm"],
        "score_isolation_forest":bs["Isolation Forest"][0],
        "alarm_isolation_forest":bs["Isolation Forest"][1],
        "score_oneclass_svm":bs["One-Class SVM"][0],
        "alarm_oneclass_svm":bs["One-Class SVM"][1],
    })
    # keep a separate index/time-domain indicator
    pred_df["sample_group"]=np.where(np.arange(len(pred_df))<len(p["healthy_test"]),"heldout_healthy","abnormal_event")

    # CIs
    block=max(5,int(round(1.0/(ms/1000.0))))  # one-second block
    ci=block_bootstrap_ci(p["y"],pred["ewma_score"],pred["alarm"],block_size=block)
    cirow={"method":"Proposed state-conditioned residual","block_size_samples":block,**ci}
    for name,(s,a) in bs.items():
        ci2=block_bootstrap_ci(p["y"],s,a,block_size=block)
        # stored later
        pass

    citable=[]
    for method,s,a in [
        ("Proposed state-conditioned residual",pred["ewma_score"],pred["alarm"]),
        ("Isolation Forest",bs["Isolation Forest"][0],bs["Isolation Forest"][1]),
        ("One-Class SVM",bs["One-Class SVM"][0],bs["One-Class SVM"][1]),
    ]:
        c=block_bootstrap_ci(p["y"],s,a,block)
        citable.append({"method":method,"block_size_samples":block,**c})
    ci_table=pd.DataFrame(citable)

    paired=[]
    for name,(s,a) in bs.items():
        d=paired_block_diff_ci(p["y"],pred["ewma_score"],pred["alarm"],s,a,block)
        paired.append({"comparison":f"Proposed minus {name}",**d})
    paired=pd.DataFrame(paired)

    # Residual contribution summary during event: standardized absolute residual median
    R=pred["residuals"]
    hN=len(p["healthy_test"])
    event_R=R[hN:]
    contribution=[]
    for j,name in enumerate(RESPONSES):
        vals=np.abs(event_R[:,j])
        contribution.append({
            "response":name,
            "median_abs_scaled_residual":float(np.nanmedian(vals)),
            "q95_abs_scaled_residual":float(np.nanquantile(vals,.95))
        })
    contribution=pd.DataFrame(contribution).sort_values("median_abs_scaled_residual",ascending=False)

    return p,det,base,pd.DataFrame(rows),ablation,seg,pred_df,ci_table,paired,contribution

if __name__=="__main__":
    p,det,base,bench,abl,seg,pred,ci,paired,contrib=primary_experiment(20,.99)
    # Do not write or redistribute record-level derived data in tracked repo.
    out=ROOT/"results"
    out.mkdir(exist_ok=True)
    bench.to_csv(out/"primary_benchmark_20ms.csv",index=False)
    abl.to_csv(out/"ablation_20ms.csv",index=False)
    seg.to_csv(out/"healthy_segment_false_alarm_20ms.csv",index=False)
    pred.to_csv(out/"primary_predictions_20ms.csv",index=False)
    ci.to_csv(out/"bootstrap_95ci_20ms.csv",index=False)
    paired.to_csv(out/"paired_block_bootstrap_differences_20ms.csv",index=False)
    contrib.to_csv(out/"event_residual_contribution_20ms.csv",index=False)
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"models"/"proposed_model_parameters.json").write_text(json.dumps(det.parameter_summary(),indent=2))
    (ROOT/"models").mkdir(exist_ok=True)
    joblib.dump(det,ROOT/"models"/"proposed_detector.joblib")
    joblib.dump(base,ROOT/"models"/"baseline_models.joblib")
    meta={
        "temporal_resolution_ms":20,
        "event_onset_s":p["onset"],"event_end_s":p["end"],
        "train_rows":len(p["train"]),"calibration_rows":len(p["calib"]),
        "heldout_healthy_rows":len(p["healthy_test"]),"event_rows":len(p["event"]),
        "calibration_quantile":.99,"ewma_alpha":.25,"persistence_samples":3,
        "context_features":CONTEXTS,"response_features":RESPONSES,
    }
    (out/"experiment_metadata.json").write_text(json.dumps(meta,indent=2))
    print("PRIMARY BENCHMARK")
    print(bench.to_string(index=False))
    print("\nABLATION")
    print(abl.to_string(index=False))
    print("\nBOOTSTRAP 95% CI")
    print(ci.to_string(index=False))
    print("\nPAIRED BLOCK-BOOTSTRAP DIFFERENCES")
    print(paired.to_string(index=False))
    print("\nEVENT RESIDUAL CONTRIBUTION")
    print(contrib.to_string(index=False))
