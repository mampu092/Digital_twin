"""Execute synthetic multihorizon, state-space, BP and randomized-trial demonstrations."""
import _bootstrap
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error
from src.common import ROOT, dump_json
from src.research.synthetic_cohort import cohort, randomized_trial
from src.research.multimodal import LongitudinalMultimodal, multitask_loss, cumulative_event_probability
from src.research.causal import estimate_ate, policy_value, safe_utility_choice
from src.research.state_space import ResearchStateSpace, AXES, H
from src.research.longitudinal import bp_summary, history_as_of


def run(epochs=30):
    torch.manual_seed(732)
    torch.set_num_threads(2)
    values, masks, time, labels, available = cohort()
    # Disjoint participants, fixed 800/200/200. Validation only selects checkpoint.
    tr, va, te = np.arange(800), np.arange(800,1000), np.arange(1000,1200)
    model = LongitudinalMultimodal()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    trace, best, best_state = [], float("inf"), None
    def slice_dict(d, ix):
        return {k:v[ix] for k,v in d.items()}
    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(np.random.default_rng(732+epoch).permutation(tr), 13):
            optimizer.zero_grad()
            out = model(slice_dict(values,ix), slice_dict(masks,ix), time[ix])
            loss,_ = multitask_loss(out, slice_dict(labels,ix), slice_dict(available,ix))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(slice_dict(values,va), slice_dict(masks,va), time[va])
            loss, components = multitask_loss(out, slice_dict(labels,va), slice_dict(available,va))
        trace.append({"epoch":epoch+1, "validation_loss":float(loss), **{k:float(v) for k,v in components.items()}})
        if float(loss) < best:
            best, best_state = float(loss), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(slice_dict(values,te), slice_dict(masks,te), time[te])
        _, components = multitask_loss(out, slice_dict(labels,te), slice_dict(available,te))
    acute_mask, bp_mask = available["acute"][te].numpy(), available["bp"][te].numpy()
    y = labels["acute"][te].numpy()[acute_mask]
    p = torch.sigmoid(out["acute"]).numpy()[acute_mask]
    bp = out["bp"].numpy()[bp_mask]*[15,10]+[120,75]
    truth = labels["bp"][te].numpy()[bp_mask]*[15,10]+[120,75]
    persistence = values["physiology"][te,-1,2:4].numpy()[bp_mask]*[15,10]+[120,75]
    persistence_mask = masks["physiology"][te,-1,2:4].numpy()[bp_mask].astype(bool)
    # Persistence comparison must use the same observed last-reading cases.
    matched = persistence_mask.all(1)
    hazard = cumulative_event_probability(out["event"]).numpy()
    result = {"provenance":"ENTIRELY SYNTHETIC; no clinical performance claim",
              "participants":{"train":800,"validation":200,"test":200},
              "epochs":epochs,"selected_epoch":int(np.argmin([r["validation_loss"] for r in trace])+1),
              "horizons":{"acute":"Next nightly step; does NOT validate minutes/hours stress recognition",
                          "bp":"Mean of seven readings in the following week",
                          "psych":"Next-step invented unitless scores; not questionnaire diagnoses",
                          "event":"Five illustrative annual intervals after landmark"},
              "acute":{"n":len(y), "roc_auc":float(roc_auc_score(y,p)),
                       "average_precision":float(average_precision_score(y,p)), "brier":float(brier_score_loss(y,p))},
              "bp":{"n":len(truth),"sbp_mae_mmhg":float(mean_absolute_error(truth[:,0],bp[:,0])),
                    "dbp_mae_mmhg":float(mean_absolute_error(truth[:,1],bp[:,1])),
                    "matched_persistence_n":int(matched.sum()),
                    "matched_model_sbp_mae":float(mean_absolute_error(truth[matched,0],bp[matched,0])),
                    "matched_persistence_sbp_mae":float(mean_absolute_error(truth[matched,0],persistence[matched,0]))},
              "survival":{"right_censoring_handled":True,"test_likelihood_loss":float(components["event"]),
                          "intervals":5,"unit":"illustrative annual intervals from landmark",
                          "mean_cumulative_probability":hazard.mean(0).tolist()},
              "psych_huber_loss":float(components["psych"]),
              "limits":"Synthetic assumptions, no external validation, no individualized uncertainty calibration, no inferred causal effects"}
    torch.save({"state_dict":best_state,"provenance":"synthetic_only","modalities":list(values),
                "history_days":21,"event_intervals":5},ROOT/"models/saved/longitudinal_synthetic.pt")
    pd.DataFrame(trace).to_csv(ROOT/"outputs/metrics/longitudinal_training.csv",index=False)
    dump_json(result,ROOT/"outputs/metrics/longitudinal_synthetic_evaluation.json")
    rng = np.random.default_rng(733)
    twin, latent, states, readings = ResearchStateSpace(), np.zeros(5), [], []
    for day in range(1,85):
        exposure = float(day >= 22)
        latent = .9*latent+np.array([.15,.08,.02,.05,.1])*exposure+rng.normal(0,.05,5)
        obs = H@latent+rng.normal(0,.2,6)
        # Deliberately never collect cortisol or hsCRP: prior-driven axes stay labeled.
        obs[[2,3]] = np.nan
        if day % 13 == 0:
            obs[:] = np.nan
        state = twin.update(obs, exposure=exposure)
        for k,axis in enumerate(AXES):
            states.append(dict(day=day,axis=axis,mean=state["mean"][k],sd=state["sd"][k],
                               measurement_supported=state["measurement_supported"][k],provenance="synthetic"))
        for asleep in [False,True]:
            sbp = 128+4*latent[3]-(12 if asleep else 0)+rng.normal(0,2)
            readings.append(dict(timestamp=(pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(days=day,hours=2 if asleep else 12)).isoformat(),
                                 sbp=sbp,dbp=80+2*latent[3]-(7 if asleep else 0)+rng.normal(0,1),
                                 asleep=asleep,quality="valid",device_validated=True,provenance="synthetic"))
    pd.DataFrame(states).to_csv(ROOT/"data/synthetic/research_state_trajectory.csv",index=False)
    bp_frame = pd.DataFrame(readings)
    bp_frame.to_csv(ROOT/"data/synthetic/research_bp_readings.csv",index=False)
    dump_json(bp_summary(bp_frame),ROOT/"outputs/metrics/synthetic_bp_summary.json")
    trial = randomized_trial(n=1600)
    # Baseline-independent centered outcome keeps the HT estimate unbiased and less noisy.
    # A fixed 130 reference is prespecified; both raw and centered estimators are reported.
    centered = trial.copy()
    centered["outcome"] -= 130
    proximal = randomized_trial(n=500, decisions=7, seed=734)
    pc = proximal.copy()
    pc["outcome"] -= 130
    reward = proximal.copy()
    reward["outcome"] = 130-reward.outcome-.5*reward.action
    causal = {"provenance":"SYNTHETIC TRIALS ONLY",
              "parallel_trial_raw":estimate_ate(trial), "parallel_trial_centered":estimate_ate(centered),
              "known_parallel_effect":-4.0,
              "mrt_proximal_centered":estimate_ate(pc),
              "mrt_effect_by_prespecified_state":{
                  "state_at_most_0.5":estimate_ate(pc.loc[pc.state<=.5]),
                  "state_above_0.5":estimate_ate(pc.loc[pc.state>.5])},
              "known_mrt_eligible_mean_effect":float(proximal.loc[proximal.eligible.eq(1),"known_effect"].mean()),
              "fixed_policy_value":policy_value(reward, (reward.state>.5).astype(int)),
              "reward_definition":"130 - observed SBP - 0.5 * action; invented utility, fixed state>0.5 policy",
              "utility_example":safe_utility_choice([
                  dict(name="no_prompt",safe=True,physiological_benefit=0,psychological_benefit=0,burden=0,risk=0),
                  dict(name="optional_social_contact_prompt",safe=True,physiological_benefit=.2,psychological_benefit=.5,burden=.2,risk=.1)]),
              "limitations":"No carryover or future treatment before proximal outcome; not a validated personalized treatment policy"}
    trial.to_csv(ROOT/"data/synthetic/parallel_trial.csv",index=False)
    proximal.to_csv(ROOT/"data/synthetic/proximal_trial.csv",index=False)
    dump_json(causal,ROOT/"outputs/metrics/synthetic_causal_evaluation.json")
    timeline = pd.DataFrame([
        dict(patient_id="SYN-ASOF",timestamp="2026-01-01T06:00:00Z",available_at="2026-01-01T06:00:00Z",
             variable="sbp",value=128,unit="mmHg",quality="valid"),
        dict(patient_id="SYN-ASOF",timestamp="2026-01-01T06:00:00Z",available_at="2026-01-03T06:00:00Z",
             variable="hs_crp",value=2,unit="mg/L",quality="valid"),
        dict(patient_id="SYN-ASOF",timestamp="2026-01-04T06:00:00Z",available_at="2026-01-04T06:00:00Z",
             variable="sbp",value=130,unit="mmHg",quality="valid")])
    visible=history_as_of(timeline,"SYN-ASOF","2026-01-02T06:00:00Z")
    dump_json({"provenance":"synthetic","input_rows":3,"visible_rows":len(visible),
               "visible_variables":visible.variable.tolist(),"delayed_assay_and_future_bp_excluded":True},
              ROOT/"outputs/metrics/asof_demonstration.json")
    print(result)
    print({"known_ate":-4,"estimated_centered":causal["parallel_trial_centered"]})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",type=int,default=30)
    args=parser.parse_args()
    if args.epochs<1:
        parser.error("epochs must be positive")
    run(args.epochs)
