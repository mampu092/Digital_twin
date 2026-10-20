"""Replay or run frozen model inference on derived feature vectors."""
import argparse
import json
from pathlib import Path
import numpy as np
from monte_carlo_digital_twin import MTL,STATIC_COLS,WEAR_COLS,PSYCH_COLS


def feature_names():
    return (STATIC_COLS+[k+'_missing' for k in STATIC_COLS]
            +WEAR_COLS+[k+'_missing' for k in WEAR_COLS]
            +['hr_now','hr_mean_4h','hr_sd_4h','sbp_now','sbp_mean_4h','rmssd_now','hour_sin','hour_cos']
            +['glucose_now','glucose_mean_4h','glucose_slope_per_step','glucose_sd_4h','ate_recent_1h']
            +PSYCH_COLS+['stress_now','rmssd_x_stress_now']+[k+'_missing' for k in PSYCH_COLS])


def predict(features,model_path):
    z = np.load(model_path)
    x = np.atleast_2d(np.asarray(features,dtype=np.float32))
    if x.shape[1] != len(z['scaler_mean']) or not np.isfinite(x).all():
        raise ValueError('Expected finite derived feature vectors matching feature_schema.json')
    m = MTL(x.shape[1])
    for k in m._params(): setattr(m,k,z[k])
    f = ((x-z['scaler_mean'])/z['scaler_scale']).astype(np.float32)
    _,sbp,stress,p = m.forward(f)
    return {'spike_probability_2h':p.tolist(),
            'sbp_2h_mmHg':(sbp*z['sbp_scale'][1]+z['sbp_scale'][0]).tolist(),
            'stress_2h':(stress*z['stress_scale'][1]+z['stress_scale'][0]).tolist()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'outputs')
    p.add_argument('--features-json',type=Path,help='JSON array of derived feature vectors, in schema order')
    args = p.parse_args()
    (args.out/'feature_schema.json').write_text(json.dumps({'feature_order':feature_names(),
        'note':'Use training_medians_run0.json for missing source values; missingness flags precede imputation. Rolling features end at decision time. Glucose and SBP are simulated in this project.'},indent=2))
    x = json.loads(args.features_json.read_text()) if args.features_json else np.load(args.out/'inference_replay.npz')['features']
    result = predict(x,args.out/'model_full_run0.npz')
    if not args.features_json:
        expected = np.load(args.out/'inference_replay.npz')['probabilities']
        np.testing.assert_allclose(result['spike_probability_2h'],expected,atol=2e-6,rtol=2e-5)
        print(f'Frozen-model replay passed for {len(expected)} held-out feature vectors.')
    else: print(json.dumps(result,indent=2))
