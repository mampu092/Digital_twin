import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, balanced_accuracy_score, brier_score_loss)


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    two = len(np.unique(y)) == 2
    return dict(n=len(y), prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(y,p)) if two else None,
        pr_auc=float(average_precision_score(y,p)) if two else None,
        sensitivity=float(tp/(tp+fn)) if tp+fn else None,
        specificity=float(tn/(tn+fp)) if tn+fp else None,
        precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        brier=float(brier_score_loss(y,p)), threshold=float(threshold),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def bootstrap(y, p, groups=None, threshold=.5, repeats=200, seed=42):
    """Percentile intervals; resample patient/profile clusters, not individual repeats."""
    y,p=np.asarray(y),np.asarray(p)
    groups=np.arange(len(y)) if groups is None else np.asarray(groups)
    unique=np.unique(groups)
    blocks={g:np.flatnonzero(groups==g) for g in unique}
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(repeats):
        ix=np.concatenate([blocks[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))==2:
            samples.append(metrics(y[ix],p[ix],threshold))
    names=["roc_auc","pr_auc","sensitivity","specificity","precision","recall","f1","balanced_accuracy","brier"]
    return {k:[float(x) for x in np.quantile([s[k] for s in samples],[.025,.975])] for k in names}


def paired_ablation(y, clinical, combined, groups, repeats=200):
    rng=np.random.default_rng(42)
    unique=np.unique(groups)
    blocks={g:np.flatnonzero(groups==g) for g in unique}
    diffs=[]
    for _ in range(repeats):
        ix=np.concatenate([blocks[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ix]))==2:
            diffs.append([roc_auc_score(y[ix],combined[ix])-roc_auc_score(y[ix],clinical[ix]),
                          average_precision_score(y[ix],combined[ix])-average_precision_score(y[ix],clinical[ix])])
    return {"roc_auc_delta":float(roc_auc_score(y,combined)-roc_auc_score(y,clinical)),
            "pr_auc_delta":float(average_precision_score(y,combined)-average_precision_score(y,clinical)),
            "delta_ci95":np.quantile(diffs,[.025,.975],axis=0).T.tolist(),
            "interpretation":"paired cluster bootstrap; internal associations, not clinical incremental utility"}
