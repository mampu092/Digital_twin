import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import monte_carlo_digital_twin as mc


def test_time_major_predictions_and_alerts():
    flat = np.array([.1,.2,.3,.4,.5,.6])  # 2 patients, 3 decision times
    mat = mc.prediction_matrix(flat,2,3)
    np.testing.assert_array_equal(mat,[[.1,.3,.5],[.2,.4,.6]])
    np.testing.assert_array_equal(mat.T.ravel(),flat)


def test_auc_ties_are_half_credit():
    assert mc.auc_score(np.array([0,1,0,1]),np.ones(4)) == .5


def test_scalar_bias_updates():
    model = mc.MTL(2)
    before = np.array([model.bs,model.bp,model.be])
    model.step(np.ones((4,2),np.float32),np.ones(4),np.ones(4),np.ones(4),1.)
    assert np.all(np.array([model.bs,model.bp,model.be]) != before)


def test_bootstrap_identity_disjoint_and_train_imputation():
    columns = list(dict.fromkeys(mc.STATIC_COLS+mc.WEAR_COLS+mc.PSYCH_COLS))
    c = pd.DataFrame({k:np.arange(100,dtype=float)+1 for k in columns})
    c['patient_id'] = [str(i) for i in range(100)]
    c.loc[::3,'rmssd'] = np.nan
    p,split,medians = mc.split_and_impute(c,500,4)
    groups = [set(p.loc[split==s,'patient_id']) for s in ['train','validation','test']]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    original = c.set_index('patient_id').loc[p.loc[split=='train','patient_id'],'rmssd']
    assert medians.rmssd == original.median()
    assert not p.rmssd.isna().any()
    assert p.rmssd_missing.sum() > 0


def test_features_never_see_future():
    x = np.arange(40,dtype=float).reshape(2,20)
    before = mc.rolling_feats(x,8,5)
    x[:,9:] = 999999
    after = mc.rolling_feats(x,8,5)
    for a,b in zip(before,after): np.testing.assert_array_equal(a,b)


def test_lead_is_within_two_hours_and_excludes_current():
    g = np.full((1,40),100.)
    g[:,20:25] = 190
    decisions = np.array([4,12,20])
    result = mc.lead_metrics(g,np.array([[.9,.9,.9]]),decisions,.5)
    assert result['alert_lead_min'] == 120
    assert result['max_lead_min'] <= 120
    assert result['crossings_detected'] == 1


def test_window_alert_confusion_counts():
    m = mc.alert_metrics(np.array([0,1,1,0]),np.array([.8,.7,.2,.1]),.5,'x_')
    assert (m['x_tp'],m['x_fp'],m['x_fn']) == (1,1,1)
    assert m['x_precision'] == m['x_recall'] == .5


def test_event_horizon_has_no_current_value_leak():
    n,t = 1,30
    sim = {'G':np.full((n,t),100.),'RM':np.ones((n,t))*40,
           'HR':np.ones((n,t))*60,'BR':np.ones((n,t))*12,'night':np.ones(t,dtype=bool)}
    sim['G'][0,15] = 190
    y,_,_ = mc.event_labels(sim)
    assert y[0,7] == 1 and y[0,6] == 0 and y[0,15] == 0


def test_saved_model_predictions_replay():
    root = Path(__file__).resolve().parents[1]/'outputs'
    if not (root/'model_full_run0.npz').exists(): return
    z = np.load(root/'model_full_run0.npz')
    replay = np.load(root/'inference_replay.npz')
    m = mc.MTL(len(z['scaler_mean']))
    for k in m._params(): setattr(m,k,z[k])
    f = ((replay['features']-z['scaler_mean'])/z['scaler_scale']).astype(np.float32)
    np.testing.assert_allclose(m.forward(f)[3],replay['probabilities'],rtol=2e-5,atol=2e-6)


def test_age_filter_includes_18_and_60(tmp_path):
    columns=list(dict.fromkeys(mc.STATIC_COLS+mc.WEAR_COLS+mc.PSYCH_COLS))
    d=pd.DataFrame({k:np.ones(50) for k in columns})
    d['patient_id']=[str(i) for i in range(50)]
    d['gender']='Female'
    d['smoking_history']='never'
    d['age']=40.
    d.loc[:3,'age']=[17,18,60,61]
    path=tmp_path/'data.csv'
    d.to_csv(path,index=False)
    raw,c=mc.load_cohort(path)
    assert len(c)==48 and c.age.min()==18 and c.age.max()==60
    assert '0' not in set(c.patient_id) and '3' not in set(c.patient_id)
    with pytest.raises(ValueError,match='outside age range'):
        mc.load_cohort(path,['0'])
    d.loc[1,'patient_id']='0'
    d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='unique'):
        mc.load_cohort(path)


def test_randomized_trial_recovers_known_linear_effect():
    risk=np.linspace(-1.5,1.5,300)
    g=np.repeat((150+10*risk)[:,None],mc.N_STEPS,axis=1)
    result=mc.run_mrt(risk,{'G':g},np.random.default_rng(88),np.ones(len(risk),dtype=bool))
    for k in ['lo','mid','hi']:
        assert result['beta_'+k]==pytest.approx(result['beta_true_'+k],abs=1e-8)
