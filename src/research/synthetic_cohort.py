"""Invented longitudinal generative mechanism. No rows from the source CSV."""
import numpy as np
import pandas as pd
import torch
from scipy.special import expit


def cohort(n=1200, seed=730, days=21):
    rng = np.random.default_rng(seed)
    age = rng.uniform(25, 75, n)
    bmi = rng.uniform(20, 38, n)
    loneliness = rng.uniform(0, 1, n)  # unitless invented exposure, NOT a UCLA scale
    physiology, psychology, assays, context = [], [], [], []
    anxiety = .7*loneliness+rng.normal(0, .1, n)
    for day in range(days):
        anxiety = .7*anxiety+.3*(.7*loneliness)+rng.normal(0, .12, n)
        activity = rng.uniform(0, 1, n)
        sleep = 7.5-1.2*anxiety+rng.normal(0, .35, n)
        sbp = 112+.35*(age-25)+.5*(bmi-25)+7*anxiety+rng.normal(0, 3, n)
        dbp = 70+.18*(age-25)+3*anxiety+rng.normal(0, 2, n)
        physiology.append(np.stack([(60+10*anxiety-3*activity-65)/10,
                                    -anxiety+rng.normal(0, .15, n),
                                    (sbp-120)/15, (dbp-75)/10,
                                    anxiety+rng.normal(0, .3, n), (sleep-7)/1.5], -1))
        psychology.append(np.stack([loneliness, anxiety, .5*anxiety+rng.normal(0, .1, n)], -1))
        assays.append(np.stack([anxiety+rng.normal(0, .3, n),
                                .3*(bmi-25)/10+rng.normal(0, .3, n),
                                (bmi-25)/10+rng.normal(0, .3, n)], -1))
        context.append(np.stack([(age-50)/20, (bmi-27)/6, activity, (sleep-7)/1.5,
                                 np.zeros(n)], -1))
    values = {k: np.stack(v, 1).astype("float32") for k,v in
              zip(["physiology", "psychology", "assays", "context"], [physiology, psychology, assays, context])}
    masks = {k: (rng.random(v.shape) > (.6 if k=="assays" else .12)).astype("float32") for k,v in values.items()}
    time = np.tile(np.stack([np.arange(days)/days, np.ones(days)/days], -1)[None], (n,1,1)).astype("float32")
    # Future labels generated separately, after the history endpoint. Label noise is independent.
    future_anxiety = .8*anxiety+.2*.7*loneliness+rng.normal(0, .18, n)
    acute = rng.binomial(1, expit(3*future_anxiety-1.3)).astype("float32")
    # The BP head forecasts the mean of seven invented readings in the next week.
    future_sbp = 112+.35*(age-25)+.5*(bmi-25)+7*future_anxiety+rng.normal(0, 8, (n,7)).mean(1)
    future_dbp = 70+.18*(age-25)+3*future_anxiety+rng.normal(0, 6, (n,7)).mean(1)
    event = np.zeros(n, dtype="float32")
    observed_bins = np.full(n, 5, dtype="float32")
    censor = rng.integers(1, 6, n)
    for i in range(n):
        for b in range(censor[i]):
            hazard = expit(-3+.035*(age[i]-50)+.04*(bmi[i]-27)+.5*anxiety[i]+.12*b)
            if rng.random() < hazard:
                event[i], observed_bins[i] = 1, b+1
                break
        else:
            observed_bins[i] = censor[i]
    labels = {"acute": acute, "bp": np.stack([(future_sbp-120)/15, (future_dbp-75)/10], -1).astype("float32"),
              "psych": np.stack([loneliness, future_anxiety, .5*future_anxiety], -1).astype("float32"),
              "event": event, "observed_bins": observed_bins}
    available = {k: (rng.random(n) > .08) for k in ["acute", "bp", "psych", "event"]}
    return ({k:torch.tensor(v) for k,v in values.items()}, {k:torch.tensor(v) for k,v in masks.items()},
            torch.tensor(time), {k:torch.tensor(v) for k,v in labels.items()},
            {k:torch.tensor(v) for k,v in available.items()})


def randomized_trial(n=800, decisions=1, seed=731):
    """Known synthetic effect -4 mmHg; reward policy effect depends on exposure.

    For decisions>1 this is an immediate proximal trial without carryover, future
    treatment or overlapping outcome windows, rather than a full clinical MRT.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        baseline = rng.normal(130, 8)
        for j in range(decisions):
            state = float(rng.uniform())
            p = .3+.4*state
            eligible = int(rng.random() > .1)
            # Eligibility precedes assignment; unavailable people receive no prompt.
            action = int(rng.random() < p) if eligible else 0
            effect = -4.0 if decisions == 1 else -2-4*state
            noise = rng.normal(0,3,7).mean() if decisions==1 else rng.normal(0,3)
            outcome = baseline+effect*action+noise
            decision = pd.Timestamp("2026-01-01", tz="UTC")+pd.Timedelta(days=j)
            rows.append(dict(patient_id=f"SYN-TRIAL-{i:04d}", decision_time=decision.isoformat(),
                             outcome_time=(decision+(pd.Timedelta(weeks=12) if decisions==1 else pd.Timedelta(hours=12))).isoformat(),
                             action=action, assignment_probability=p, eligible=eligible,
                             outcome=outcome, state=state, known_effect=effect,
                             outcome_definition="final-week mean SBP at week 12" if decisions==1 else "12-hour proximal SBP",
                             provenance="synthetic"))
    return pd.DataFrame(rows)
