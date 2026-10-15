"""Rebuild figures from local (unpublished) per-row predictions and public aggregate results.

The generated figure files are scientific illustrations; time-indexed predictions
must remain in the local workspace under the repository's data rights policy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'/'generated'
OUT.mkdir(parents=True,exist_ok=True)

def save(fig, stem):
    fig.savefig(OUT/f'{stem}.png',dpi=600,bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.svg',bbox_inches='tight')
    plt.close(fig)

def main():
    bench=pd.read_csv(ROOT/'results'/'primary_benchmark_20ms.csv')
    fig,ax=plt.subplots(figsize=(7.16,3.5))
    bench.set_index('method')[['f1','auroc','false_alarm_rate']].plot(kind='bar',ax=ax)
    ax.set_title('Strict-causal powertrain anomaly benchmark')
    ax.set_ylim(0,1.05); ax.set_ylabel('Metric value'); ax.grid(axis='y',alpha=0.25)
    ax.tick_params(axis='x',labelrotation=15)
    fig.tight_layout();save(fig,'benchmark')

    prediction_path=ROOT/'results'/'primary_predictions_20ms.csv'
    if not prediction_path.exists():
        print('No per-row predictions available locally; saved aggregate benchmark figure only.')
        return
    pred=pd.read_csv(prediction_path)
    labels=pred['ground_truth'].to_numpy(int)
    methods={
        'Proposed residual':'score_proposed',
        'Isolation Forest':'score_isolation_forest',
        'One-Class SVM':'score_oneclass_svm',
    }
    fig,ax=plt.subplots(figsize=(3.5,3.1))
    for name,col in methods.items():
        fpr,tpr,_=roc_curve(labels,pred[col]);ax.plot(fpr,tpr,label=name)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves')
    ax.legend(frameon=False,fontsize=7);ax.grid(alpha=0.2)
    save(fig,'roc')
    fig,ax=plt.subplots(figsize=(3.5,3.1))
    for name,col in methods.items():
        precision,recall,_=precision_recall_curve(labels,pred[col])
        ax.plot(recall,precision,label=name)
    ax.set(xlabel='Recall',ylabel='Precision',title='Precision-recall curves')
    ax.legend(frameon=False,fontsize=7);ax.grid(alpha=0.2)
    save(fig,'precision_recall')

if __name__ == '__main__':
    main()
