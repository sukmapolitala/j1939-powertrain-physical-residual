from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
R=Path('/mnt/data/j1939-powertrain-physical-residual')
d=pd.read_csv(R/'results/primary_predictions_20ms.csv'); y=d.ground_truth.to_numpy(dtype=int)
methods={
 'Proposed state-conditioned residual':('score_proposed','alarm_proposed'),
 'Isolation Forest':('score_isolation_forest','alarm_isolation_forest'),
 'One-Class SVM':('score_oneclass_svm','alarm_oneclass_svm')
}
score={k:d[v[0]].to_numpy(float) for k,v in methods.items()}
alarm={k:d[v[1]].to_numpy(int) for k,v in methods.items()}

def f1_far(yy,aa):
 tp=int(((yy==1)&(aa==1)).sum()); fp=int(((yy==0)&(aa==1)).sum()); fn=int(((yy==1)&(aa==0)).sum()); tn=int(((yy==0)&(aa==0)).sum())
 f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
 far=fp/(fp+tn) if (fp+tn) else float('nan')
 return f1,far
rng=np.random.default_rng(20261015); block=50; n=len(y); starts=np.arange(n-block+1); nblocks=int(np.ceil(n/block))
store={m:{z:[] for z in ['f1','false_alarm_rate','auroc','auprc']} for m in methods}
pairs={other:{z:[] for z in ['delta_f1','delta_far']} for other in ['Isolation Forest','One-Class SVM']}
for _ in range(1000):
 starts_draw=rng.choice(starts,size=nblocks,replace=True)
 idx=np.concatenate([np.arange(s,s+block) for s in starts_draw])[:n]
 yy=y[idx]
 if not (0 in yy and 1 in yy): continue
 cur={}
 for m in methods:
  f,far=f1_far(yy,alarm[m][idx])
  cur[m]=(f,far)
  store[m]['f1'].append(f);store[m]['false_alarm_rate'].append(far)
  store[m]['auroc'].append(roc_auc_score(yy,score[m][idx]));store[m]['auprc'].append(average_precision_score(yy,score[m][idx]))
 for b in pairs:
  pairs[b]['delta_f1'].append(cur['Proposed state-conditioned residual'][0]-cur[b][0]);pairs[b]['delta_far'].append(cur['Proposed state-conditioned residual'][1]-cur[b][1])
rows=[]
for m,metrics in store.items():
 r={'method':m,'bootstrap_repeats':len(metrics['f1']),'block_size_samples':block}
 for z,values in metrics.items():
  r[f'{z}_ci_low']=float(np.quantile(values,.025));r[f'{z}_ci_high']=float(np.quantile(values,.975))
 rows.append(r)
pd.DataFrame(rows).to_csv(R/'results/bootstrap_1000_95ci_20ms.csv',index=False)
rows=[]
for b,ds in pairs.items():
 r={'comparison':f'Proposed minus {b}','bootstrap_repeats':len(ds['delta_f1'])}
 for z,values in ds.items():
  r[f'{z}_ci_low']=float(np.quantile(values,.025));r[f'{z}_ci_high']=float(np.quantile(values,.975))
 rows.append(r)
pd.DataFrame(rows).to_csv(R/'results/paired_bootstrap_1000_20ms.csv',index=False)
print(pd.read_csv(R/'results/bootstrap_1000_95ci_20ms.csv').to_string(index=False))
print(pd.read_csv(R/'results/paired_bootstrap_1000_20ms.csv').to_string(index=False))
