import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def fit_calibrator(y, probabilities):
    return LogisticRegression(C=1e6, random_state=42).fit(logit(probabilities),y)


def calibrate(calibrator, probabilities):
    return calibrator.predict_proba(logit(probabilities))[:,1]


def select_threshold(y,p):
    thresholds=np.unique(np.quantile(p,np.linspace(.02,.98,150)))
    return float(max(thresholds,key=lambda t:balanced_accuracy_score(y,p>=t)))
