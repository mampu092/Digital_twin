"""Five-axis linear Gaussian demonstrator; coefficients are invented, not fitted physiology."""
import numpy as np

AXES = ["ANS", "HPA", "immune", "vascular", "behavior"]
OBSERVATIONS = ["hr_z", "negative_log_rmssd_z", "cortisol_z", "hs_crp_z", "sbp_z", "sleep_deficit_z"]
H = np.array([[1,0,0,0,0], [.8,0,0,0,0], [0,1,0,0,0],
              [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], dtype=float)


class ResearchStateSpace:
    def __init__(self):
        self.mean = np.zeros(5)
        self.covariance = np.eye(5)
        self.observed_axes = np.zeros(5, dtype=bool)

    def update(self, values, elapsed_days=1, exposure=0, action=0, context=0, provenance="synthetic"):
        if provenance != "synthetic":
            raise ValueError("Unidentified biological coefficients: this demonstrator accepts synthetic inputs only")
        y = np.asarray(values, dtype=float)
        if y.shape != (6,) or np.isinf(y).any():
            raise ValueError("Six standardized measurements; NaN represents missing")
        if not np.isfinite([elapsed_days, exposure, action, context]).all() or elapsed_days <= 0:
            raise ValueError("Finite covariates and positive elapsed time required")
        # Diagonal dynamics keep unsupported axes prior-driven. Process noise grows with gaps.
        decay = np.array([.8, .9, .97, .95, .85])**elapsed_days
        transition = np.diag(decay)
        drive = (np.array([.12,.08,.02,.04,.1])*exposure -
                 np.array([.06,.03,0,.02,.06])*action + np.array([.03,0,0,.01,.05])*context)
        self.mean = transition@self.mean+drive
        self.covariance = transition@self.covariance@transition.T+np.eye(5)*.08*elapsed_days
        mask = np.isfinite(y)
        if mask.any():
            h = H[mask]
            r = np.eye(mask.sum())*.2
            gain = np.linalg.solve(h@self.covariance@h.T+r, h@self.covariance).T
            self.mean += gain@(y[mask]-h@self.mean)
            identity = np.eye(5)-gain@h
            # Joseph covariance update preserves positive semidefiniteness.
            self.covariance = identity@self.covariance@identity.T+gain@r@gain.T
            self.observed_axes |= (np.abs(h).sum(axis=0) > 0)
        return {"axes": AXES, "mean": self.mean.tolist(),
                "sd": np.sqrt(np.diag(self.covariance)).tolist(),
                "measurement_supported": self.observed_axes.tolist(),
                "interpretation": "Synthetic latent coordinates; HPA/immune priors are not measured biology"}
