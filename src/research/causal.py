"""Auditable one-decision causal estimators, with participant-cluster uncertainty.

Calling these functions cannot establish exchangeability/consistency. Supplied
data must justify those assumptions. The shipped execution uses synthetic trials.
"""
import numpy as np
from src.research.longitudinal import validate_trial_log


def _interval(contributions, groups, repeats=500, seed=42):
    if repeats < 100:
        raise ValueError("Use at least 100 cluster bootstrap repetitions")
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError("At least 10 independent participants required for this bootstrap")
    blocks = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    draws = [float(contributions[np.concatenate([blocks[j] for j in
              rng.integers(0, len(blocks), len(blocks))])].mean()) for _ in range(repeats)]
    return np.quantile(draws, [.025, .975]).tolist()


def estimate_ate(frame, repeats=500):
    data = validate_trial_log(frame)
    a, p, y = (data[c].to_numpy(float) for c in ["action", "assignment_probability", "outcome"])
    contribution = a*y/p-(1-a)*y/(1-p)
    return {"estimate": float(contribution.mean()),
            "ci95": _interval(contribution, data.patient_id, repeats),
            "n_eligible_decisions": len(data), "n_participants": data.patient_id.nunique(),
            "min_assignment_probability": float(np.minimum(p, 1-p).min()),
            "max_inverse_weight": float(np.maximum(1/p, 1/(1-p)).max()),
            "estimand": "Decision-weighted eligible proximal effect; participant ATE only for one decision each",
            "method": "Horvitz-Thompson; no clipping; participant-cluster percentile bootstrap",
            "assumptions": "Known randomized probabilities, consistency, no interference, nonoverlapping outcomes; no missing outcomes"}


def policy_value(frame, policy_actions, repeats=500):
    """Policy must be fixed independently of evaluation outcomes.

    policy_actions is indexed exactly like frame. Reward is supplied in outcome
    and must be prespecified; a BP reduction score is not an observed clinical benefit.
    """
    if len(policy_actions) != len(frame):
        raise ValueError("One policy action per input decision required")
    raw = frame.copy()
    raw["_policy"] = np.asarray(policy_actions)
    data = validate_trial_log(raw)
    if not data._policy.isin([0, 1]).all():
        raise ValueError("Binary policy required")
    a, p, y = (data[c].to_numpy(float) for c in ["action", "assignment_probability", "outcome"])
    weight = (a == data._policy.to_numpy())/np.where(a == 1, p, 1-p)
    if weight.sum() == 0:
        raise ValueError("No logged action supports this policy")
    contribution = weight*y
    return {"ipw_value": float(contribution.mean()),
            "ci95": _interval(contribution, data.patient_id, repeats),
            "matched_decisions": int((weight > 0).sum()), "eligible_decisions": len(data),
            "effective_sample_size": float(weight.sum()**2/(weight@weight)),
            "maximum_weight": float(weight.max()),
            "method": "Unnormalized IPW policy value, fixed policy, participant-cluster bootstrap",
            "scope": "One-decision reward under logging history; not cumulative dynamic-policy value"}


def safe_utility_choice(actions):
    """Argmax over caller-eligible actions under explicitly supplied assumptions."""
    usable = []
    for action in actions:
        if not action.get("safe", False):
            continue
        values = [action.get(k) for k in ["physiological_benefit", "psychological_benefit", "burden", "risk"]]
        if any(v is None for v in values) or not np.isfinite(values).all():
            continue
        utility = values[0]+values[1]-values[2]-values[3]
        usable.append((float(utility), str(action["name"])))
    if not usable:
        return {"selected": None, "status": "no_supported_safe_action"}
    utility, name = max(usable)
    return {"selected": name, "assumed_utility": utility,
            "status": "synthetic_assumption_comparison_not_treatment_recommendation"}
