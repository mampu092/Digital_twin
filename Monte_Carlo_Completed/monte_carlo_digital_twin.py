"""Completed Monte Carlo research prototype. All longitudinal outcomes are synthetic.
Adapted from the user's original_reference.py. See README.md for assumptions.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from threadpoolctl import threadpool_limits

N_DAYS, DT_MIN = 7, 15
STEPS_DAY = 96
N_STEPS = N_DAYS * STEPS_DAY
HORIZON, WARMUP, DECIDE_EVERY = 8, 96, 8
LAMBDAS = dict(l_BP=1.0, l_psych=0.3, l_event=1.0, l_cal=1.0)
STATIC_COLS = ['age','gender_male','bmi','HbA1c_level','hypertension',
               'heart_disease','diabetes','smoke_current','kidney_disease',
               'stroke','blood_glucose_level','bad_physical_days','gen_health']
WEAR_COLS = ['nremhr','rmssd','spo2','full_sleep_breathing_rate',
             'sleep_points_percentage','sleep_hours','nightly_temperature']
PSYCH_COLS = ['TENSE/ANXIOUS','TIRED','stress_score','bad_mental_days']

def simulate_paths(pat, rng):
    """Monte-Carlo draw of glucose / HR / SBP / RMSSD / stress trajectories.

    Mechanistic kernels:
      glucose: AR(1) mean-reversion + meal shocks, damped by insulin
               effectiveness eps_i = exp(-0.35*(HbA1c_i-5)); spikes grow as
               insulin resistance grows (higher HbA1c, diabetes).
      HR:      circadian baseline + activity bursts + stress coupling.
      SBP:     OU anchored on hypertension status, coupled to stress & HR.
      RMSSD:   OU anchored on wearable resting HRV, suppressed by stress.
    """
    N, T = len(pat), N_STEPS
    t_idx = np.arange(T)
    hour_of_day = (t_idx * DT_MIN / 60.0) % 24
    ho = hour_of_day[None, :]

    hba1c = pat["HbA1c_level"].values.astype(float)
    bmi   = pat["bmi"].values.astype(float)
    age   = pat["age"].values.astype(float)
    hyper = pat["hypertension"].values.astype(float)
    diab  = pat["diabetes"].values.astype(float)
    base_hr = pat["nremhr"].values.astype(float)
    base_rm = pat["rmssd"].values.astype(float)
    base_br = pat["full_sleep_breathing_rate"].values.astype(float)
    base_st = np.clip(pat["stress_score"].values.astype(float) / 100.0, 0.05, 0.95)

    insulin_eff = np.exp(-0.35 * np.clip(hba1c - 5.0, 0, 4))
    mu_g = 85 + 25 * diab + 0.25 * bmi + 0.3 * (age - 40) * diab
    kappa_g = 0.035 * insulin_eff + 0.005
    sigma_g = 0.55 + 0.9 * (1 - insulin_eff)

    meal_times = [8.0, 13.0, 19.5]
    meal_mask = np.zeros((N, T), dtype=bool)
    for mt in meal_times:
        jitter = rng.normal(0, 0.5, size=(N, 1))
        mtime = (mt + jitter) % 24
        dist = np.abs(((ho - mtime + 12) % 24) - 12)
        meal_mask |= (dist < 0.25)
    snack_rate = 0.35 + 0.4 * diab
    snacks = rng.random((N, T)) < (snack_rate[:, None] * (DT_MIN / (16 * 60)))
    eat = (meal_mask | snacks) & (rng.random((N, T)) < 0.85)

    G = np.zeros((N, T));   G[:, 0] = pat["blood_glucose_level"].values.astype(float)
    HR = np.zeros((N, T));  HR[:, 0] = base_hr + 8
    SBP = np.zeros((N, T)); SBP[:, 0] = 112 + 16 * hyper + 0.15 * (age - 40)
    RM = np.zeros((N, T));  RM[:, 0] = base_rm
    ST = np.zeros((N, T));  ST[:, 0] = base_st

    circ = np.sin((ho - 3.0) / 24 * 2 * np.pi)
    hr_circ = 62 + 9 * circ + 0.06 * (age[:, None] - 40)   # (N, T)
    activity = np.clip(rng.lognormal(-1.2, 0.9, size=(N, T)), 0, 4) * (circ > -0.2)

    for t in range(1, T):
        eps = rng.standard_normal(N)
        shock = eat[:, t] * rng.gamma(2.0, 8 + 18 * (1 - insulin_eff)) * (rng.random(N) < 0.9)
        newG = G[:, t-1] + kappa_g * (mu_g - G[:, t-1]) * 3 + sigma_g * eps + shock * 0.9
        G[:, t] = np.clip(newG, np.maximum(G[:, t-1] - 12, 55), np.minimum(G[:, t-1] + 45, 420))

        stress_drive = 0.02 * diab * rng.random(N)
        ST[:, t] = np.clip(ST[:, t-1] + 0.15 * (base_st - ST[:, t-1])
                           + 0.02 * eps + stress_drive, 0, 1)
        HR[:, t] = np.clip(hr_circ[:, t] + 6 * activity[:, t] + 12 * ST[:, t]
                           + rng.normal(0, 1.5, N) + 4 * hyper, 38, 190)
        # OU around a stress-modulated anchor: stress raises the effective target
        # but the reversion keeps SBP bounded (no runaway drift)
        anchor_eff = 112 + 16 * hyper + 0.15 * (age - 40) + 8 * ST[:, t]
        SBP[:, t] = np.clip(SBP[:, t-1] + 0.06 * (anchor_eff - SBP[:, t-1])
                            + rng.normal(0, 1.5, N), 85, 210)
        RM[:, t] = np.clip(RM[:, t-1] + 0.08 * (base_rm - RM[:, t-1])
                           - 12 * (ST[:, t] - ST[:, t-1]) + rng.normal(0, 2.0, N), 8, 120)

    night = (hour_of_day >= 1) & (hour_of_day <= 5)
    return dict(G=G, HR=HR, SBP=SBP, RM=RM, ST=ST, BR=base_br[:, None] + 6 * ST,
                hour=hour_of_day, night=night, eat=eat)

# -----------------------------------------------------------------------------
# 3. ADVERSE-EVENT GROUND TRUTH
# -----------------------------------------------------------------------------
def event_labels(sim):
    G, RM, HR, BR, night = sim["G"], sim["RM"], sim["HR"], sim["BR"], sim["night"]
    T = G.shape[1]
    fut_max = None
    for k in range(1, HORIZON + 1):
        shifted = np.full_like(G, -np.inf)
        shifted[:, :T-k] = G[:, k:]
        fut_max = shifted if fut_max is None else np.maximum(fut_max, shifted)
    # Y[i, t] = 1 : glucose spike within the next 2 h (crosses 180 from below
    #               while currently below, OR rises >= 70 mg/dL over the window)
    Y_spike = ((fut_max >= 180) & (G < 180)) | ((fut_max - G) >= 70)
    night_cols = np.where(night)[0]
    nhr = HR[:, night_cols].mean(axis=1)
    nrm = RM[:, night_cols].mean(axis=1)
    nbr = BR[:, night_cols].mean(axis=1)
    cardio_anom = ((nhr > 68) & (nrm < 28) & (nbr > 17)).astype(float)
    return Y_spike.astype(float), cardio_anom, dict(nhr=nhr, nrm=nrm, nbr=nbr)

# -----------------------------------------------------------------------------
# 4. FEATURES from history H_{i,t}  (leak-free: only steps <= t)
# -----------------------------------------------------------------------------
def rolling_feats(x, t, w):
    w = min(w, t + 1)
    seg = x[:, t - w + 1: t + 1]
    return seg.mean(axis=1), (seg[:, -1] - seg[:, 0]) / w, seg.std(axis=1), seg[:, -1]

def build_dataset(pat, sim, Y, decision_idx):
    X, y_ev, y_sbp, y_ps, ids = [], [], [], [], []
    for t in decision_idx:
        gm, gs, gd, g0 = rolling_feats(sim["G"], t, 16)     # last 4 h of glucose
        hm, hs, hd, h0 = rolling_feats(sim["HR"], t, 16)
        sm, ss, sd, s0 = rolling_feats(sim["SBP"], t, 16)
        rm_, _, _, r0  = rolling_feats(sim["RM"], t, 16)
        tm, _, _, t0   = rolling_feats(sim["ST"], t, 16)
        ate_recent = sim["eat"][:, max(0, t - 4): t + 1].any(axis=1).astype(float)
        n_pat = len(g0)
        hsin = np.full(n_pat, np.sin(2 * np.pi * sim["hour"][t] / 24))
        hcos = np.full(n_pat, np.cos(2 * np.pi * sim["hour"][t] / 24))
        dyn = np.column_stack([g0, gm, gs, gd, h0, hm, hd, s0, sm, r0, t0,
                               ate_recent, r0 * t0, hsin, hcos])
        X.append(dyn)
        y_ev.append(Y[:, t])                                  # event in (t, t+2h]
        y_sbp.append(sim["SBP"][:, t + HORIZON])              # SBP at t+2h
        y_ps.append(sim["ST"][:, t + HORIZON])                # stress at t+2h
        ids.append(np.arange(len(pat)))
    X = np.vstack(X).astype(np.float32)
    y_ev = np.concatenate(y_ev).astype(np.float32)
    y_sbp = np.concatenate(y_sbp).astype(np.float32)
    y_ps = np.concatenate(y_ps).astype(np.float32)
    pat_idx = np.concatenate(ids)
    Xs = pat[STATIC_COLS].values.astype(np.float32)[pat_idx]
    Xp = pat[PSYCH_COLS].values.astype(np.float32)[pat_idx]
    return X, Xs, Xp, y_ev, y_sbp, y_ps, pat_idx

# -----------------------------------------------------------------------------
# 5. MODEL B: multitask network (shared hidden layer, 3 heads) trained by Adam
#    L = l1*Huber(SBP) + l2*MSE(stress) + l3*BCE(event) + l4*(calibration gap)^2
# -----------------------------------------------------------------------------
def huber(z, d=8.0):
    a = np.abs(z)
    return np.where(a <= d, 0.5 * z**2, d * (a - 0.5 * d))

class MTL:
    def __init__(self, d_in, h=12, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.normal(0, 1/np.sqrt(d_in), (d_in, h))).astype(np.float32)
        self.b1 = np.zeros(h, np.float32)
        self.Ws = rng.normal(0, 1/np.sqrt(h), h).astype(np.float32)
        self.Wp = rng.normal(0, 1/np.sqrt(h), h).astype(np.float32)
        self.We = rng.normal(0, 1/np.sqrt(h), h).astype(np.float32)
        self.bs = np.float32(0); self.bp = np.float32(0); self.be = np.float32(0)
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return dict(W1=self.W1, b1=self.b1, Ws=self.Ws, bs=self.bs,
                    Wp=self.Wp, bp=self.bp, We=self.We, be=self.be)

    def forward(self, X):
        H = np.maximum(X @ self.W1 + self.b1, 0)
        return H, H @ self.Ws + self.bs, H @ self.Wp + self.bp, \
               1 / (1 + np.exp(-np.clip(H @ self.We + self.be, -35, 35)))

    def step(self, X, y_ev, y_sbp, y_ps, y_mean, lr=0.02, lam=LAMBDAS):
        n = len(X)
        H, sbp_h, ps_h, p = self.forward(X)
        p_c = np.clip(p, 1e-6, 1 - 1e-6)
        L = (lam["l_BP"] * huber(sbp_h - y_sbp).mean()
             + lam["l_psych"] * ((ps_h - y_ps) ** 2).mean()
             + lam["l_event"] * -(y_ev * np.log(p_c) + (1 - y_ev) * np.log(1 - p_c)).mean()
             + lam["l_cal"] * (p.mean() - y_mean) ** 2)
        # gradients (BCE+sigmoid: dL/dlogit = (p-y)/n ; calibration term chain-ruled)
        dL_dsbp = (lam["l_BP"] * np.clip(sbp_h - y_sbp, -8, 8)) / n
        dL_dps  = (lam["l_psych"] * 2 * (ps_h - y_ps)) / n
        dL_dp   = ((p - y_ev) / n
                   + lam["l_cal"] * 2 * (p.mean() - y_mean) * (p * (1 - p)) / n)
        dH = (dL_dsbp[:, None] * self.Ws.T + dL_dps[:, None] * self.Wp.T
              + dL_dp[:, None] * self.We.T) * (H > 0)
        grads = dict(W1=X.T @ dH, b1=dH.sum(0),
                     Ws=H.T @ dL_dsbp, bs=dL_dsbp.sum(),
                     Wp=H.T @ dL_dps, bp=dL_dps.sum(),
                     We=H.T @ dL_dp, be=dL_dp.sum())
        P = self._params()
        b1_, b2_, eps_ = 0.9, 0.999, 1e-8
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1_ * self.m[k] + (1 - b1_) * g
            self.v[k] = b2_ * self.v[k] + (1 - b2_) * g * g
            mh = self.m[k] / (1 - b1_ ** self.t)
            vh = self.v[k] / (1 - b2_ ** self.t)
            setattr(self, k, P[k] - (lr * mh / (np.sqrt(vh) + eps_)).astype(np.float32))
        return float(L)

def auc_score(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')

# -----------------------------------------------------------------------------
# 6. MICRO-RANDOMIZED TRIAL inside the twin
#    daily decision points; availability I~Bern(.8); A~Bern(.5) = activity prompt
#    TRUE built-in effect: prompt reduces next-day glucose peak by
#    beta_true(s) = 4 + 8*(s+1.5)/3  mg/dL   (s = standardized Model-A risk)
#    -> ~4 mg/dL for low-risk, ~12 mg/dL for high-risk. Always beneficial.
# -----------------------------------------------------------------------------
def load_cohort(path, patient_ids=None):
    raw = pd.read_csv(path, dtype={'patient_id': str})
    required = set(STATIC_COLS + WEAR_COLS + PSYCH_COLS) - {'gender_male','smoke_current'}
    missing = required.union({'patient_id','gender','smoking_history'}) - set(raw)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if raw.patient_id.isna().any() or raw.patient_id.duplicated().any():
        raise ValueError('patient_id must be present and unique at patient-summary grain')
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    c = raw[raw.age.between(18,60,inclusive='both')].copy().reset_index(drop=True)
    if patient_ids:
        wanted = set(patient_ids)
        if wanted - set(c.patient_id):
            raise ValueError('Requested patient IDs missing or outside age range 18-60')
        c = c[c.patient_id.isin(wanted)].reset_index(drop=True)
    if len(c) < 30:
        raise ValueError('At least 30 eligible distinct patients required for three-way evaluation')
    # Never fold unknown gender or smoking into a false clinical assertion.
    c['gender_male'] = c.gender.map({'Male':1.0,'Female':0.0,'Other':np.nan})
    c['smoke_current'] = c.smoking_history.map({'current':1.0,'ever':np.nan,
                          'former':0.0,'never':0.0,'not current':0.0,'No Info':np.nan})
    return raw, c


def privacy_audit(c):
    qi = pd.DataFrame({'age':c.age,'gender':c.gender,'bmi':c.bmi.round(0),
                       'smoking':c.smoking_history})
    k = qi.groupby(list(qi),dropna=False).size()
    return {'rows':len(c),'equivalence_classes':len(k),'min_k':int(k.min()),
            'median_class_k':float(k.median()),
            'unique_record_pct':float(k[k == 1].sum()/len(c)*100),
            'classes_k_le_3_pct':float((k <= 3).mean()*100),
            'record_k_le_3_pct':float(k[k <= 3].sum()/len(c)*100),
            'compliance_certified':False,
            'source_license_and_synthetic_status':'Unverified: supplied CSV has no provenance documentation'}


def audit_data(raw, c):
    triplet = c[['nremhr','rmssd','full_sleep_breathing_rate']].dropna()
    repeats = triplet.value_counts()
    return {'source_rows':len(raw),'source_columns':len(raw.columns),'cohort_rows':len(c),
            'excluded_age_rows':int((~raw.age.between(18,60)).sum()),
            'unique_patient_ids':int(c.patient_id.nunique()),'duplicate_ids':int(c.patient_id.duplicated().sum()),
            'age_min':float(c.age.min()),'age_max':float(c.age.max()),
            'grain':'One summary row per patient; no timestamps or measured CGM trajectories',
            'wearable_profile_reuse':{'complete_hr_hrv_breathing_rows':len(triplet),
                                      'distinct_triplets':len(repeats),
                                      'largest_repeat_count':int(repeats.max()) if len(repeats) else 0,
                                      'rows_in_repeated_triplets':int(repeats[repeats>1].sum()),
                                      'interpretation':'Repeated summary values across source patient IDs; original wearable-user linkage and independence unverified'},
            'wearable_unique_nonmissing_values':{k:int(c[k].nunique()) for k in WEAR_COLS},
            'observed_coverage':{k:float(c[k].notna().mean()) for k in WEAR_COLS+PSYCH_COLS},
            'age_histogram':{str(int(k)):int(v) for k,v in c.age.value_counts().sort_index().items()},
            'condition_prevalence':{k:float(c[k].mean()) for k in ['diabetes','hypertension','heart_disease','stroke','kidney_disease']},
            'privacy':privacy_audit(c)}


def split_and_impute(c, n, seed):
    p = c.sample(n, replace=True,random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed + 90000)
    unique = rng.permutation(p.patient_id.unique())
    if len(unique) < 15:
        raise ValueError('Too few unique bootstrap identities for evaluation')
    a,b = int(.6*len(unique)),int(.8*len(unique))
    membership = {v:s for s,vs in zip(['train','validation','test'],[unique[:a],unique[a:b],unique[b:]]) for v in vs}
    split = p.patient_id.map(membership).to_numpy()
    sets = [set(p.loc[split == s,'patient_id']) for s in ['train','validation','test']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    columns = list(dict.fromkeys(STATIC_COLS+WEAR_COLS+PSYCH_COLS))
    absent = p[columns].isna().astype(float).add_suffix('_missing')
    medians = p.loc[split == 'train',columns].median()
    if medians.isna().any():
        raise ValueError(f'No train observations for {medians[medians.isna()].index.tolist()}')
    p[columns] = p[columns].fillna(medians)
    return pd.concat([p,absent],axis=1),split,medians


def run_mrt(risk_std, sim, rng, test_mask):
    # Treatment-centered regression of randomized next-day outcomes.
    # Cluster inference by bootstrap trajectory, preserving repeated daily decisions.
    from scipy.stats import t as student_t
    risk = risk_std[test_mask]
    G = sim['G'][test_mask]
    n = len(risk)
    q = np.quantile(risk,[1/3,2/3])
    group = np.searchsorted(q,risk,side='right')
    true = 4 + 8*(risk+1.5)/3
    rows = []
    for day in range(N_DAYS-1):
        available = rng.random(n) < .8
        treatment = rng.random(n) < .5
        baseline = G[:,(day+1)*STEPS_DAY:(day+2)*STEPS_DAY].max(axis=1)
        outcome = baseline - true*treatment
        previous_peak = G[:,day*STEPS_DAY:(day+1)*STEPS_DAY].max(axis=1)
        rows.append(pd.DataFrame({'cluster':np.arange(n),'group':group,'risk':risk,
                                  'treatment':treatment.astype(float),'outcome':outcome,
                                  'previous_peak':previous_peak,'day':day,
                                  'available':available,'true_effect':true}))
    trial = pd.concat(rows,ignore_index=True)
    trial = trial[trial.available]
    results = {}
    for j,name in enumerate(['lo','mid','hi']):
        z = trial[trial.group == j]
        risk_centered = z.risk.to_numpy()-z.risk.mean()
        treatment_centered = z.treatment.to_numpy()-.5
        design = np.column_stack([np.ones(len(z)),treatment_centered,risk_centered,
                                  treatment_centered*risk_centered,
                                  (z.previous_peak.to_numpy()-z.previous_peak.mean())/50,
                                  *[(z.day.to_numpy()==d).astype(float) for d in range(1,N_DAYS-1)]])
        inverse = np.linalg.pinv(design.T@design)
        coef = inverse@design.T@z.outcome.to_numpy()
        residual = z.outcome.to_numpy()-design@coef
        clusters = z.cluster.to_numpy()
        cluster_scores = np.stack([(design[clusters==i]*residual[clusters==i,None]).sum(0)
                                   for i in np.unique(clusters)])
        m = len(cluster_scores)
        adjustment = m/max(m-1,1)*(len(z)-1)/max(len(z)-design.shape[1],1)
        covariance = adjustment*inverse@(cluster_scores.T@cluster_scores)@inverse
        estimate = -coef[1]
        se = float(np.sqrt(max(covariance[1,1],0)))
        radius = float(student_t.ppf(.975,m-1))*se
        truth = z.true_effect.mean()
        results.update({f'beta_{name}':estimate,f'beta_se_{name}':se,
                        f'beta_true_{name}':truth,f'beta_low_{name}':estimate-radius,
                        f'beta_high_{name}':estimate+radius,
                        f'beta_covered_{name}':float(estimate-radius<=truth<=estimate+radius)})
    return results


def train_mtl(Ftr, targets, seed, epochs):
    m = MTL(Ftr.shape[1],seed=seed)
    ev,sbp,stress = targets
    m.sbp_scale = (float(sbp.mean()),float(sbp.std()+1e-8))
    m.ps_scale = (float(stress.mean()),float(stress.std()+1e-8))
    # Correct prior initialization and ensure all scalar biases update.
    prior = np.clip(ev.mean(),1e-5,1-1e-5)
    m.be = np.float32(np.log(prior/(1-prior)))
    sub = np.random.default_rng(seed).choice(len(Ftr),min(30000,len(Ftr)),replace=False)
    s = (sbp-m.sbp_scale[0])/m.sbp_scale[1]
    p = (stress-m.ps_scale[0])/m.ps_scale[1]
    for _ in range(epochs):
        m.step(Ftr[sub],ev[sub],s[sub],p[sub],float(ev[sub].mean()),lr=.015)
    return m


def metrics(y,p,prefix=''):
    bins = np.minimum((p*10).astype(int),9)
    ece = sum(abs(p[bins==b].mean()-y[bins==b].mean())*(bins==b).mean()
              for b in range(10) if (bins==b).any())
    return {prefix+'auc':auc_score(y,p),prefix+'ap':float(average_precision_score(y,p)) if y.sum() else np.nan,
            prefix+'brier':float(np.mean((p-y)**2)),prefix+'ece':float(ece)}


def choose_threshold(y,p):
    if not y.sum():
        return 1.0
    precision,recall,thresholds = precision_recall_curve(y,p)
    f2 = 5*precision[:-1]*recall[:-1]/(4*precision[:-1]+recall[:-1]+1e-12)
    return float(thresholds[np.argmax(f2)])


def alert_metrics(y,p,threshold,prefix):
    alert = p >= threshold
    tp = int((alert & (y==1)).sum())
    return {prefix+'threshold':float(threshold),prefix+'n_alerts':int(alert.sum()),
            prefix+'tp':tp,prefix+'fp':int((alert & (y==0)).sum()),prefix+'fn':int((~alert & (y==1)).sum()),
            prefix+'precision':tp/int(alert.sum()) if alert.sum() else np.nan,
            prefix+'recall':tp/int(y.sum()) if y.sum() else np.nan,
            prefix+'alerts_per_100_windows':float(alert.mean()*100)}


def prediction_matrix(flat,n_pat,n_times):
    """build_dataset is time-major: all patients at t0, then all at t1."""
    return np.asarray(flat).reshape(n_times,n_pat).T


def lead_metrics(G,p,decisions,threshold):
    leads,eligible,detected = [],0,0
    for i in range(len(G)):
        crossings = np.flatnonzero((G[i,1:] >= 180)&(G[i,:-1] < 180))+1
        for cross in crossings:
            prior = (decisions < cross)&(decisions >= cross-HORIZON)
            if not prior.any():
                continue
            eligible += 1
            times = decisions[prior & (p[i] >= threshold)]
            if len(times):
                detected += 1
                leads.append(float((cross-times.min())*DT_MIN))
    return {'crossings_eligible':eligible,'crossings_detected':detected,
            'crossing_recall':detected/eligible if eligible else np.nan,
            'alert_lead_min':float(np.mean(leads)) if leads else np.nan,
            'max_lead_min':float(max(leads)) if leads else np.nan}


def one_run(c,run_id,args):
    seed = args.seed+run_id
    rng = np.random.default_rng(seed+1000)
    pat,split,medians = split_and_impute(c,args.patients,seed)
    tr,va,te = [split == x for x in ['train','validation','test']]
    sim = simulate_paths(pat,rng)
    Y,cardio,_ = event_labels(sim)
    decisions = np.arange(WARMUP,N_STEPS-HORIZON,DECIDE_EVERY)
    weekly = ((Y[:,decisions].max(1)>0)|(cardio>0)).astype(float)
    scA = StandardScaler().fit(pat.loc[tr,STATIC_COLS])
    mA = LogisticRegression(max_iter=800).fit(scA.transform(pat.loc[tr,STATIC_COLS]),weekly[tr])
    risk = mA.predict_proba(scA.transform(pat[STATIC_COLS]))[:,1]
    risk_std = np.clip((risk-risk[tr].mean())/(risk[tr].std()+1e-8),-1.5,1.5)
    xd,xs,xp,y,sbp,stress,idx = build_dataset(pat,sim,Y,decisions)
    train,valid,test = tr[idx],va[idx],te[idx]
    ws = pat[WEAR_COLS+[k+'_missing' for k in WEAR_COLS]].to_numpy(np.float32)[idx]
    # Isolate stress to the final contextual block. CGM is a separate simulated modality.
    wd = xd[:,[4,5,6,7,8,9,13,14]]
    cgm = xd[:,[0,1,2,3,11]]
    missing_clinical = pat[[k+'_missing' for k in STATIC_COLS]].to_numpy(np.float32)[idx]
    clinical = np.hstack([xs,missing_clinical])
    wearable = np.hstack([clinical,ws,wd])
    fusion = np.hstack([wearable,cgm])
    full = np.hstack([fusion,xp,xd[:,[10,12]],
                      pat[[k+'_missing' for k in PSYCH_COLS]].to_numpy(np.float32)[idx]])
    blocks = {'static':clinical,'wearable_no_cgm':wearable,'fusion':fusion,'full':full}
    result = {'run':run_id,'seed':seed,'auc_A_weekly':auc_score(weekly[te],risk[te]),
              'spike_rate':float(y[test].mean()),'cardio_rate':float(cardio[te].mean()),
              'any_event_rate':float(weekly[te].mean()),'test_windows':int(test.sum()),
              'test_positive_windows':int(y[test].sum()),'test_patients':int(te.sum()),
              'train_unique_ids':int(pat.loc[tr,'patient_id'].nunique()),
              'validation_unique_ids':int(pat.loc[va,'patient_id'].nunique()),
              'test_unique_ids':int(pat.loc[te,'patient_id'].nunique()),'identity_overlap':0}
    model_bundle = None
    for name,F in blocks.items():
        scaler = StandardScaler().fit(F[train])
        f = scaler.transform(F).astype(np.float32)
        m = train_mtl(f[train],(y[train],sbp[train],stress[train]),seed,args.epochs)
        _,sbp_pred,stress_pred,p = m.forward(f)
        result.update(metrics(y[test],p[test],name+'_'))
        if name == 'full':
            threshold = choose_threshold(y[valid],p[valid])
            result.update(alert_metrics(y[test],p[test],threshold,'selected_'))
            result.update(alert_metrics(y[test],p[test],.30,'fixed_'))
            result['mae_sbp_mmHg'] = float(np.abs(sbp[test]-(sbp_pred[test]*m.sbp_scale[1]+m.sbp_scale[0])).mean())
            result['mae_stress'] = float(np.abs(stress[test]-(stress_pred[test]*m.ps_scale[1]+m.ps_scale[0])).mean())
            pmat = prediction_matrix(p,len(pat),len(decisions))
            result.update(lead_metrics(sim['G'][te],pmat[te],decisions,threshold))
            model_bundle = (m,scaler,F,p,pmat,threshold)
    result['delta_wearable'] = result['wearable_no_cgm_auc']-result['static_auc']
    result['delta_cgm'] = result['fusion_auc']-result['wearable_no_cgm_auc']
    result['delta_psych'] = result['full_auc']-result['fusion_auc']
    result['brier_constant_baseline'] = float(np.mean((y[test]-y[train].mean())**2))
    result.update(run_mrt(risk_std,sim,rng,te))
    examples = []
    if run_id == 0:
        m,scaler,F,p,pmat,threshold = model_bundle
        # Selection uses baseline risk only; no retrospective event/alert selection.
        ordered = np.flatnonzero(te)[np.argsort(risk[te])]
        selected = np.unique(ordered[np.linspace(0,len(ordered)-1,5).astype(int)])
        for j,i in enumerate(selected):
            profile = {'virtual_patient_id':f'VT-{j+1:03}',
                       'age_band':f'{max(18,int(pat.age.iloc[i]//5*5))}-{min(60,int(pat.age.iloc[i]//5*5+4))}',
                       'diabetes':int(pat.diabetes.iloc[i]),'hypertension':int(pat.hypertension.iloc[i]),
                       'weekly_risk':float(risk[i]),'split':'held-out test',
                       'selection':'baseline-risk quantiles, independent of future events',
                       'static_origin':'source-derived; age coarsened; original ID excluded'}
            series = {'hours':(np.arange(N_STEPS)*DT_MIN/60).tolist(),
                      **{k:sim[k][i].tolist() for k in ['G','HR','SBP','RM','ST']}}
            valid_y = Y[i].astype(float)
            valid_y[-HORIZON:] = np.nan
            series['Y_spike'] = [None if not np.isfinite(x) else int(x) for x in valid_y]
            decision = {'idx':decisions.tolist(),'hours':(decisions*DT_MIN/60).tolist(),
                        'p':pmat[i].tolist(),'y':Y[i,decisions].astype(int).tolist(),
                        'threshold':threshold}
            examples.append({'profile':profile,'series':series,'decision':decision})
        np.savez_compressed(args.out/'model_full_run0.npz',**m._params(),
                            scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                            sbp_scale=m.sbp_scale,stress_scale=m.ps_scale)
        # Validation-only calibration plot bins and replay vectors, no patient identifiers.
        np.savez_compressed(args.out/'heldout_predictions_run0.npz',y=y[test],p=p[test])
        replay = np.flatnonzero(test)[::max(1,int(test.sum()/100))]
        np.savez_compressed(args.out/'inference_replay.npz',features=F[replay],probabilities=p[replay])
        (args.out/'training_medians_run0.json').write_text(json.dumps(medians.to_dict(),indent=2))
    return result,examples


def safe_json(value):
    if isinstance(value,dict): return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe_json(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'outputs')
    parser.add_argument('--runs',type=int,default=40)
    parser.add_argument('--patients',type=int,default=1500)
    parser.add_argument('--epochs',type=int,default=180)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--patient-ids',nargs='+',help='Optional explicit source IDs; ages still restricted to 18-60')
    args = parser.parse_args()
    if args.runs<1 or args.patients<60 or args.epochs<1:
        parser.error('runs >=1, patients >=60, epochs >=1 required')
    args.out.mkdir(parents=True,exist_ok=True)
    start = time.time()
    raw,c = load_cohort(args.data,args.patient_ids)
    audit = audit_data(raw,c)
    (args.out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(f'Eligible cohort: {len(c):,}; {args.runs} runs x {args.patients} bootstrap trajectories',flush=True)
    results,examples = [],[]
    with threadpool_limits(limits=1):
        for r in range(args.runs):
            result,ex = one_run(c,r,args)
            results.append(result)
            examples.extend(ex)
            pd.DataFrame(results).to_csv(args.out/'mc_run_metrics.csv',index=False)
            print(f'Run {r+1}/{args.runs}: full AUC={result["full_auc"]:.3f}, AP={result["full_ap"]:.3f}, recall={result["selected_recall"]:.3f} [{time.time()-start:.0f}s]',flush=True)
    R = pd.DataFrame(results)
    summary = {'cohort_n':len(c),'mc_runs':args.runs,'patients_per_run':args.patients,
               'days':N_DAYS,'dt_minutes':DT_MIN,'horizon_hours':2,'burn_in_hours':24,
               'interval_definition':'2.5th-97.5th percentiles across simulated runs; not clinical confidence intervals',
               'metrics':{col:{'mean':float(R[col].mean()),'low':float(R[col].quantile(.025)),
                               'high':float(R[col].quantile(.975)),'valid_runs':int(R[col].notna().sum())}
                          for col in R.select_dtypes('number') if col not in ['run','seed']},
               'privacy':audit['privacy'],'runtime_seconds':time.time()-start}
    config = {'runs':args.runs,'patients':args.patients,'epochs':args.epochs,'seed':args.seed,
              'data_filename':args.data.name,'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'source_status':'User supplied; open-source license, original linkage and synthetic EHR status unverified',
              'patient_subset_requested':bool(args.patient_ids),
              'label':'max G in (t,t+2h] >=180 while G(t)<180, OR max G-G(t)>=70; synthetic mg/dL',
              'evaluation':'60/20/20 grouped original IDs; bootstrap identities never cross splits; scalers and medians train only',
              'threshold':'Max F2 on validation only. Fixed 0.30 also reported.',
              'weekly_target':'Any modeled spike decision window after burn-in OR rule-defined weekly cardiovascular anomaly',
              'mrt':'Treatment-centered regression with risk interaction, prior-day peak and day covariates; trajectory-cluster robust SE; imposed effects, not clinical treatment evidence',
              'simulation':'Original AR mean-reversion, circadian HR and OU-style SBP/HRV kernels retained; not clinically calibrated',
              'limitations':['No measured longitudinal labels','CGM and SBP are simulated','No raw-data upload',
                             'No verified legal compliance','No physiological parameter-uncertainty calibration',
                             'Independent re-simulation and bootstrap variation do not measure external clinical validity']}
    for filename,value in [('summary.json',summary),('provenance.json',config),
                           ('example_patient_full.json',max(examples,key=lambda p:p['profile']['weekly_risk'])),
                           ('example_patient.json',max(examples,key=lambda p:p['profile']['weekly_risk'])),
                           ('dashboard_data.json',{'summary':summary,'audit':audit,'patients':examples,'provenance':config})]:
        (args.out/filename).write_text(json.dumps(safe_json(value),indent=2,allow_nan=False),encoding='utf-8')
    print(f'Completed in {time.time()-start:.1f}s. Outputs: {args.out.resolve()}',flush=True)


if __name__ == '__main__':
    main()
