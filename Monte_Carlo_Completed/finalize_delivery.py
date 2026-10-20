"""Regenerate presentation outputs and independently check saved results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from monte_carlo_digital_twin import load_cohort,audit_data
from predict import predict,feature_names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--patient-ids',nargs='+')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'outputs')
    args=parser.parse_args()
    out=args.out.resolve()
    S=json.loads((out/'summary.json').read_text())
    D=json.loads((out/'dashboard_data.json').read_text())
    P=json.loads((out/'provenance.json').read_text())
    if args.data:
        assert hashlib.sha256(args.data.read_bytes()).hexdigest()==P['data_sha256']
        raw,c=load_cohort(args.data,args.patient_ids)
        assert len(c)==S['cohort_n'],'Pass the same --patient-ids used in the run'
        A=audit_data(raw,c)
        (out/'data_audit.json').write_text(json.dumps(A,indent=2))
        D['audit']=A
    # Keep coarse age bands within the requested cohort boundaries.
    for ex in D['patients']:
        lo,hi=map(int,ex['profile']['age_band'].split('-'))
        ex['profile']['age_band']=f'{max(18,lo)}-{min(60,hi)}'
        assert ex['profile']['split']=='held-out test'
        assert len(ex['series']['G'])==672
        assert len(ex['decision']['p'])==len(ex['decision']['idx'])
        assert min(ex['decision']['hours'])>=24
        assert np.all((np.asarray(ex['decision']['p'])>=0)&(np.asarray(ex['decision']['p'])<=1))
        assert all(v is None for v in ex['series']['Y_spike'][-8:])
    example=max(D['patients'],key=lambda x:x['profile']['weekly_risk'])
    for name in ['example_patient_full.json','example_patient.json']:
        (out/name).write_text(json.dumps(example,indent=2,allow_nan=False))
    (out/'dashboard_data.json').write_text(json.dumps(D,indent=2,allow_nan=False))
    assert '"patient_id":' not in (out/'dashboard_data.json').read_text()
    R=pd.read_csv(out/'mc_run_metrics.csv')
    assert len(R)==S['mc_runs'] and R.seed.nunique()==len(R)
    assert R.identity_overlap.eq(0).all()
    assert R.max_lead_min.dropna().le(120).all()
    for key,value in S['metrics'].items():
        actual=R[key].dropna()
        assert value['valid_runs']==len(actual)
        if len(actual):
            np.testing.assert_allclose([value['mean'],value['low'],value['high']],
                                       [actual.mean(),actual.quantile(.025),actual.quantile(.975)],rtol=1e-9,atol=1e-9)
    for prefix in ['selected_','fixed_']:
        assert ((R[prefix+'tp']+R[prefix+'fp'])==R[prefix+'n_alerts']).all()
        assert ((R[prefix+'tp']+R[prefix+'fn'])==R.test_positive_windows).all()
    np.testing.assert_allclose(R.delta_wearable+R.delta_cgm+R.delta_psych,R.full_auc-R.static_auc,atol=1e-12)
    replay=np.load(out/'inference_replay.npz')
    frozen=predict(replay['features'],out/'model_full_run0.npz')
    np.testing.assert_allclose(frozen['spike_probability_2h'],replay['probabilities'],rtol=2e-5,atol=2e-6)
    assert len(feature_names())==replay['features'].shape[1]
    (out/'feature_schema.json').write_text(json.dumps({'feature_order':feature_names(),
        'note':'Train-impute source values; retain missingness flags. Features use history through decision time. CGM and BP are simulated.'},indent=2))
    from make_figures import main as figures
    from build_results import main as report
    figures(out)
    report(out)
    tests=subprocess.run([sys.executable,'-m','pytest',str(Path(__file__).parent/'tests'),'-q'],capture_output=True,text=True)
    print(tests.stdout.strip())
    assert tests.returncode==0,tests.stdout+tests.stderr
    env=os.environ.copy()
    env['TWIN_OUTPUT_DIR']=str(out)
    ui_code="""from streamlit.testing.v1 import AppTest
import json
a=AppTest.from_file(PATH,default_timeout=40).run()
assert not a.exception
a.selectbox[0].select(a.selectbox[0].options[0]).run()
a.checkbox[0].check().run()
a.slider[0].set_value(120).run()
a.slider[1].set_value(.5).run()
assert not a.exception
print(json.dumps({'exceptions':len(a.exception),'patient_options':len(a.selectbox[0].options),'checks':['initial render','patient selection','retrospective toggle','replay time','threshold change']}))
""".replace('PATH',repr(str(Path(__file__).parent/'dashboard.py')))
    ui=subprocess.run([sys.executable,'-c',ui_code],env=env,capture_output=True,text=True)
    assert ui.returncode==0,ui.stdout+ui.stderr
    ui_result=json.loads(ui.stdout.strip().splitlines()[-1])
    verification={'completed_runs':len(R),'eligible_cohort':S['cohort_n'],
                  'summary_metrics_reconciled':len(S['metrics']),'summary_recomputed_from_csv':True,
                  'alert_numerators_and_denominators_reconciled':True,'patient_identity_overlap':0,
                  'max_lead_minutes':float(R.max_lead_min.max()),'feature_count':len(feature_names()),
                  'frozen_inference_replay_vectors':len(replay['features']),
                  'tests':tests.stdout.strip(),'functional_dashboard':ui_result,
                  'source_sha256':P['data_sha256'],'clinical_validation':False,
                  'source_licensing_and_anonymization_verified':False}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
