"""Robust patient-relative deviations with a fixed enrollment reference window."""
import numpy as np
from src.data.loader import SENSORS
from src.data.simulation import NOISE


class PatientAnomalyDetector:
    def __init__(self,baseline_days=7,threshold=2.5):
        self.baseline_days=baseline_days
        self.threshold=threshold
        self.reference=None

    def fit(self,history):
        if len(history)<self.baseline_days:
            return False
        base=history[:self.baseline_days]
        centers,scales={},{}
        for i,c in enumerate(SENSORS):
            values=np.array([r.get(c) for r in base if r.get(c) is not None],dtype=float)
            values=values[np.isfinite(values)]
            if len(values)>=4:
                med=float(np.median(values))
                centers[c]=med
                scales[c]=float(max(1.4826*np.median(np.abs(values-med)),NOISE[i]))
        self.reference={"center":centers,"scale":scales}
        return len(centers)>=3

    def score(self,record):
        if not self.reference or len(self.reference["center"])<3:
            return dict(status="learning_baseline",score=None,flag=False,z={},coverage=0)
        z={c:float((record[c]-center)/self.reference["scale"][c])
           for c,center in self.reference["center"].items() if record.get(c) is not None}
        if len(z)<3:
            return dict(status="insufficient_data",score=None,flag=False,z=z,coverage=len(z)/len(SENSORS))
        # Average of largest two absolute deviations; either direction can be abnormal.
        score=float(np.mean(sorted(map(abs,z.values()))[-2:]))
        return dict(status="review_simulated_deviation" if score>=self.threshold else "within_reference",
                    score=score,flag=score>=self.threshold,z=z,coverage=len(z)/len(SENSORS))
