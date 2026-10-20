"""
DIGITAL TWIN MONTE-CARLO SIMULATION
Cohort: patients aged 18-60 from FINAL_dataset_with_fitbit.csv
        (fused static EHR stream + Fitbit wearable stream)

Implements the mathematical framework:
  Model A  (clinical baseline)        : r_i = sigma(w' x_i)  -> weekly adverse-event risk
  Model B  (physiological deterior.)  : SBP_hat_{i,t+D} = f_theta(H_{i,t})          (Huber)
                                        p_{i,D} = P(Y_{i,t:t+D}=1 | H_{i,t})        (log-loss)
                                        multitask: L = l1*L_BP + l2*L_psych + l3*L_event + l4*L_cal
  MRT      (micro-randomized trial)   : A_{i,t} ~ Bernoulli(0.5), availability I_{i,t}
                                        proximal causal effect beta(s) = E[Y_{t+D}(1)-Y_{t+D}(0) | S_t=s, I_t=1]
  Value decomposition                 : DeltaPerformance = Clinical + Wearable + Psych

All randomness is Monte-Carlo: R independent runs (patient bootstrap + re-simulated
trajectories). Outputs = metric distributions (mean +/- 95% MC interval).
"""

import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 0. CONFIG
# -----------------------------------------------------------------------------
DATA_PATH    = "/Users/apple/Desktop/Digi_twin.ipnyb/FINAL_dataset_with_fitbit.csv"
OUT          = "/Users/apple/.zcode/workspace/default/digital_twin_mc/outputs"
N_MC         = 40      # Monte-Carlo runs
N_PAT        = 1500    # patients sampled per run (bootstrap of the 18-60 cohort)
N_DAYS       = 7       # simulated days per patient
DT_MIN       = 15      # time resolution (minutes)
STEPS_DAY    = 24 * 60 // DT_MIN
N_STEPS      = N_DAYS * STEPS_DAY
HORIZON      = 2 * 60 // DT_MIN          # 2-hour prediction horizon (8 steps)
WARMUP       = 12                        # burn-in steps before decision points
DECIDE_EVERY = 8                         # decision-point cadence (2 h)
LAMBDAS      = dict(l_BP=1.0, l_psych=0.3, l_event=1.0, l_cal=1.0)

STATIC_COLS = ["age", "gender_male", "bmi", "HbA1c_level", "hypertension",
               "heart_disease", "diabetes", "smoke_current", "sleep_hours",
               "bad_mental_days", "bad_physical_days", "gen_health",
               "kidney_disease", "stroke", "physical_activity",
               "nremhr", "rmssd", "spo2", "full_sleep_breathing_rate",
               "stress_score", "sleep_points_percentage"]
PSYCH_COLS = ["TENSE/ANXIOUS", "TIRED", "stress_score", "bad_mental_days"]

# -----------------------------------------------------------------------------
# 1. LOAD + COHORT
# -----------------------------------------------------------------------------
def load_cohort():
    df = pd.read_csv(DATA_PATH)
    c = df[(df.age >= 18) & (df.age <= 60)].copy().reset_index(drop=True)
    c["gender_male"] = (c.gender == "Male").astype(float)
    c["smoke_current"] = c.smoking_history.isin(["current", "ever"]).astype(float)
    wear_cols = ["nightly_temperature", "nremhr", "rmssd", "spo2",
                 "full_sleep_breathing_rate", "stress_score", "sleep_points_percentage"]
    for w in wear_cols:
        c[f"{w}_miss"] = c[w].isna().astype(float)
        c[w] = c[w].fillna(c[w].median())
    for p in ["TENSE/ANXIOUS", "TIRED", "ENTERTAINMENT"]:
        c[p] = pd.to_numeric(c[p], errors="coerce").fillna(0.0)
    return c

# -----------------------------------------------------------------------------
# 2. DYNAMIC TWIN: 7-day physiology at 15-min resolution (vectorised over patients)
# -----------------------------------------------------------------------------
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
               1 / (1 + np.exp(-(H @ self.We + self.be)))

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
            P[k] -= (lr * mh / (np.sqrt(vh) + eps_)).astype(np.float32)
        return float(L)

def auc_score(y, p):
    o = np.argsort(p)
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# -----------------------------------------------------------------------------
# 6. MICRO-RANDOMIZED TRIAL inside the twin
#    daily decision points; availability I~Bern(.8); A~Bern(.5) = activity prompt
#    TRUE built-in effect: prompt reduces next-day glucose peak by
#    beta_true(s) = 4 + 8*(s+1.5)/3  mg/dL   (s = standardized Model-A risk)
#    -> ~4 mg/dL for low-risk, ~12 mg/dL for high-risk. Always beneficial.
# -----------------------------------------------------------------------------
def run_mrt(risk_std, sim, rng):
    N = len(risk_std)
    rows = []
    for d in range(N_DAYS - 1):
        I = rng.random(N) < 0.8
        A = (rng.random(N) < 0.5) & I
        d0, d1 = (d + 1) * STEPS_DAY, (d + 2) * STEPS_DAY
        peak_next = sim["G"][:, d0:d1].max(axis=1) - (4 + 8 * (risk_std + 1.5) / 3) * A
        rows.append(pd.DataFrame(dict(day=d, risk=np.round(risk_std, 2),
                                      I=I.astype(int), A=A.astype(int), peak=peak_next)))
    m = pd.concat(rows)
    m = m[m.I == 1]
    qs = np.quantile(m.risk, [1/3, 2/3])
    strata = [m.risk < qs[0], (m.risk >= qs[0]) & (m.risk < qs[1]), m.risk >= qs[1]]
    est, ses, ns = [], [], []
    for s in strata:
        g1 = m.peak[s & (m.A == 1)]; g0 = m.peak[s & (m.A == 0)]
        est.append(g0.mean() - g1.mean())
        ses.append(np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0)))
        ns.append(int(len(g1) + len(g0)))
    return np.array(est), np.array(ses), np.array(ns)

# -----------------------------------------------------------------------------
# 7. PRIVACY AUDIT (quasi-identifier uniqueness / k-anonymity)
# -----------------------------------------------------------------------------
def privacy_audit(cohort):
    qi = pd.DataFrame(dict(age=cohort.age, gender=cohort.gender,
                           bmi=cohort.bmi.round(0), smoke=cohort.smoking_history))
    k = qi.value_counts()
    return dict(rows=int(len(qi)),
                unique_QI_rows_pct=round(float((k == 1).mean() * 100), 2),
                median_k=float(k.median()),
                pct_equivalence_classes_k_le_3=round(float((k <= 3).mean() * 100), 2),
                min_k=int(k.min()))

# -----------------------------------------------------------------------------
# 8. ONE MONTE-CARLO RUN
# -----------------------------------------------------------------------------
def train_mtl(Ftr, ytr, seed, epochs=180):
    """ytr = (event, sbp, stress); regression targets z-scored internally so the
    Huber/MSE heads and the event head share a comparable optimisation scale."""
    model = MTL(Ftr.shape[1], seed=seed)
    ym = float(ytr[0].mean())
    mu_s, sd_s = float(ytr[1].mean()), float(ytr[1].std() + 1e-8)
    mu_p, sd_p = float(ytr[2].mean()), float(ytr[2].std() + 1e-8)
    sub = np.arange(len(Ftr))
    if len(Ftr) > 30000:
        sub = np.random.default_rng(seed).choice(len(Ftr), 30000, replace=False)
    y1 = (ytr[1] - mu_s) / sd_s
    y2 = (ytr[2] - mu_p) / sd_p
    for _ in range(epochs):
        model.step(Ftr[sub], ytr[0][sub], y1[sub], y2[sub], ym, lr=0.03)
    model.sbp_scale = (mu_s, sd_s)
    model.ps_scale = (mu_p, sd_p)
    return model

def one_run(cohort, run_id, choose_example=False):
    rng = np.random.default_rng(1000 + run_id)
    pat = cohort.sample(N_PAT, replace=True, random_state=run_id).reset_index(drop=True)
    sim = simulate_paths(pat, rng)
    Y_spike, cardio_anom, _ = event_labels(sim)
    any_event = ((Y_spike.max(axis=1) > 0) | (cardio_anom > 0)).astype(float)
    tr_pat = rng.random(len(pat)) < 0.7
    te_pat = ~tr_pat

    # ---- Model A: static clinical baseline
    XA = pat[STATIC_COLS].values.astype(np.float64)
    scA = StandardScaler().fit(XA[tr_pat])
    mA = LogisticRegression(max_iter=800, C=1.0).fit(scA.transform(XA[tr_pat]), any_event[tr_pat])
    rA = mA.predict_proba(scA.transform(XA))[:, 1]
    auc_A = auc_score(any_event[te_pat], rA[te_pat])
    risk_std = np.clip((rA - rA.mean()) / (rA.std() + 1e-9), -1.5, 1.5)

    # ---- Model B dataset (2-hourly decision points after warmup)
    decision_idx = list(range(WARMUP, N_STEPS - HORIZON, DECIDE_EVERY))
    Xd, Xs, Xp, y_ev, y_sbp, y_ps, pat_idx = build_dataset(pat, sim, Y_spike, decision_idx)
    tr_mask = tr_pat[pat_idx]; te_mask = te_pat[pat_idx]

    Fall = np.hstack([Xs, Xd, Xp]).astype(np.float32)
    Fall = ((Fall - Fall[tr_mask].mean(0)) / (Fall[tr_mask].std(0) + 1e-6)).astype(np.float32)
    D_static = Xs.shape[1]; D_dyn = Xd.shape[1]
    F_blocks = {"static": Fall[:, :D_static],
                "static+wearable": Fall[:, :D_static + D_dyn],
                "full": Fall}

    aucs, extras, models = {}, {}, {}
    for name, F in F_blocks.items():
        model = train_mtl(F[tr_mask], (y_ev[tr_mask], y_sbp[tr_mask], y_ps[tr_mask]), seed=run_id)
        _, sbp_h_z, ps_h_z, p_te = model.forward(F[te_mask])
        sbp_h = sbp_h_z * model.sbp_scale[1] + model.sbp_scale[0]
        aucs[name] = auc_score(y_ev[te_mask], p_te)
        extras[name] = (p_te, sbp_h, ps_h_z)
        models[name] = model

    p_te, sbp_h, _ = extras["full"]
    # alert matrix needs predictions at EVERY decision point for every patient
    p_all = models["full"].forward(Fall)[3]
    brier = ((p_te - y_ev[te_mask]) ** 2).mean()
    mae_sbp = np.abs(sbp_h - y_sbp[te_mask]).mean()
    bins = np.clip((p_te * 10).astype(int), 0, 9)
    ece = sum(np.abs(p_te[bins == b].mean() - y_ev[te_mask][bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())

    # alert quality (threshold 0.30): lead time before glucose crossings of 180,
    # plus precision/recall of the alert against 2-h spike labels
    thr = 0.30
    p_mat = p_all.reshape(len(pat), -1)
    alert = np.zeros((len(pat), N_STEPS), dtype=bool)
    for j, t in enumerate(decision_idx):
        alert[:, t] = p_mat[:, j] > thr
    G = sim["G"]
    leads, n_cross = [], 0
    for i in range(len(pat)):
        cross = np.where((G[i] >= 180) & (np.r_[False, G[i, :-1] < 180]))[0]
        n_cross += len(cross)
        for c in cross:
            w = alert[i, max(0, c - 16):c]
            if w.any():
                leads.append((len(w) - np.where(w)[0][0]) * DT_MIN)
    lead = float(np.mean(leads)) if leads else np.nan
    a_at = alert[:, decision_idx].reshape(-1)
    y_at = np.concatenate([Y_spike[:, t] for t in decision_idx])
    alert_precision = float(y_at[a_at].mean()) if a_at.any() else np.nan
    alert_recall = float(a_at[y_at == 1].mean()) if (y_at == 1).any() else np.nan

    beta_hat, beta_se, beta_n = run_mrt(risk_std, sim, rng)

    res = dict(run=run_id,
               auc_A_weekly=auc_A,
               auc_B_static=aucs["static"],
               auc_B_static_wearable=aucs["static+wearable"],
               auc_B_full=aucs["full"],
               brier=brier, ece=ece, mae_sbp=mgdl(mae_sbp),
               spike_rate=float(Y_spike.mean()),
               cardio_rate=float(cardio_anom.mean()),
               any_event_rate=float(any_event.mean()),
               alert_lead_min=lead, n_alerts=int(alert.sum()),
               alert_precision=alert_precision, alert_recall=alert_recall,
               n_glucose_crossings=int(n_cross),
               beta_lo=beta_hat[0], beta_mid=beta_hat[1], beta_hi=beta_hat[2],
               beta_se_lo=beta_se[0], beta_se_mid=beta_se[1], beta_se_hi=beta_se[2])

    traj = None
    if choose_example:
        # prefer a patient with at least one alert AND one true spike (informative dashboard)
        has_alert = alert.any(axis=1)
        has_spike = Y_spike.max(axis=1) > 0
        cand = np.where(has_alert & has_spike &
                        (pat.diabetes == 1).values & (rA > 0.2) & (rA < 0.9))[0]
        if not len(cand):
            cand = np.where(has_alert & has_spike)[0]
        if not len(cand):
            cand = np.where(has_alert)[0]
        if not len(cand):
            cand = np.arange(len(pat))
        # most eventful patient: most spike windows, tie-break on Model-A risk
        n_spikes = Y_spike.sum(axis=1)
        order = np.lexsort((-rA[cand], -n_spikes[cand]))
        i_ex = int(cand[order[0]])
        prof = pat.iloc[i_ex]
        res["example_patient"] = dict(
            patient_id=int(prof.patient_id), age=float(prof.age), gender=prof.gender,
            bmi=float(prof.bmi), HbA1c=float(prof.HbA1c_level), diabetes=int(prof.diabetes),
            hypertension=int(prof.hypertension), riskA=float(rA[i_ex]),
            glucose0=float(prof.blood_glucose_level), sleep_hours=float(prof.sleep_hours),
            nremhr=float(prof.nremhr), rmssd=float(prof.rmssd),
            stress=float(prof.stress_score), heart_disease=int(prof.heart_disease))
        traj = dict(hours=(np.arange(N_STEPS) * DT_MIN / 60).tolist(),
                    G=G[i_ex].tolist(), HR=sim["HR"][i_ex].tolist(),
                    SBP=sim["SBP"][i_ex].tolist(), RM=sim["RM"][i_ex].tolist(),
                    ST=sim["ST"][i_ex].tolist(),
                    alert=alert[i_ex].astype(int).tolist(),
                    Y_spike=Y_spike[i_ex].astype(int).tolist())
    return res, traj

def mgdl(x):
    return float(x)

# -----------------------------------------------------------------------------
# 9. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    cohort = load_cohort()
    print(f"Cohort 18-60: n={len(cohort)}", flush=True)
    audit = privacy_audit(cohort)
    print("Privacy audit (raw quasi-identifiers):", audit, flush=True)

    results, traj_out = [], None
    for r in range(N_MC):
        res, traj = one_run(cohort, r, choose_example=(r == 0))
        if traj is not None:
            traj_out = traj
            with open(f"{OUT}/example_patient.json", "w") as f:
                json.dump(dict(profile=res.pop("example_patient"), traj=traj), f)
        results.append(res)
        if (r + 1) % 5 == 0 or r == 0:
            print(f"  run {r+1:>2}/{N_MC}  AUC_full={res['auc_B_full']:.3f}  "
                  f"beta=({res['beta_lo']:.1f},{res['beta_mid']:.1f},{res['beta_hi']:.1f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    R = pd.DataFrame(results)
    R.to_csv(f"{OUT}/mc_run_metrics.csv", index=False)

    def ci(col, fmt="{:.3f} [{:.3f}, {:.3f}]"):
        v = R[col].dropna()
        return fmt.format(v.mean(), v.quantile(.025), v.quantile(.975))

    summary = dict(
        cohort_n=int(len(cohort)), mc_runs=N_MC, patients_per_run=N_PAT,
        horizon_hours=HORIZON * DT_MIN / 60,
        spike_label_rate_2h_windows=ci("spike_rate"),
        cardio_anomaly_weekly_rate=ci("cardio_rate"),
        any_adverse_event_week_rate=ci("any_event_rate"),
        modelA_auc_weekly_event=ci("auc_A_weekly"),
        modelB_auc_static_EHR_only=ci("auc_B_static"),
        modelB_auc_EHR_plus_wearable=ci("auc_B_static_wearable"),
        modelB_auc_full_with_psych=ci("auc_B_full"),
        delta_wearable=ci("auc_B_static_wearable").split()[0],
        brier_score=ci("brier"), calibration_ece=ci("ece"),
        sbp_2h_mae_mgdl=ci("mae_sbp", "{:.1f} [{:.1f}, {:.1f}]"),
        alert_lead_min=f"{R.alert_lead_min.mean():.0f} (sd {R.alert_lead_min.std():.0f})",
        alert_precision=ci("alert_precision"),
        alert_recall=ci("alert_recall"),
        glucose_crossings_per_patient_week=f"{(R.n_glucose_crossings / N_PAT).mean():.2f}",
        proximal_effect_beta_lo_risk_mgdl=ci("beta_lo", "{:.1f} [{:.1f}, {:.1f}]"),
        proximal_effect_beta_mid_risk_mgdl=ci("beta_mid", "{:.1f} [{:.1f}, {:.1f}]"),
        proximal_effect_beta_hi_risk_mgdl=ci("beta_hi", "{:.1f} [{:.1f}, {:.1f}]"),
        privacy_audit_raw=audit)
    print(json.dumps(summary, indent=2))
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    R.to_csv(f"{OUT}/mc_run_metrics.csv", index=False)
    print(f"Done in {time.time()-t0:.0f}s -> {OUT}")
