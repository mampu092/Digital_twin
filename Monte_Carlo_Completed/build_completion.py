"""Preserve the supplied physiology and NumPy multitask model; repair its pipeline."""
from pathlib import Path

root = Path(__file__).parent
src = (root / 'original_reference.py').read_text(encoding='utf-8')
body = src[src.index('def simulate_paths'):src.index('# -----------------------------------------------------------------------------\n# 8. ONE MONTE-CARLO RUN')]
body = body.replace('1 / (1 + np.exp(-(H @ self.We + self.be)))', '1 / (1 + np.exp(-np.clip(H @ self.We + self.be, -35, 35)))')
body = body.replace('P[k] -= (lr * mh / (np.sqrt(vh) + eps_)).astype(np.float32)', 'setattr(self, k, P[k] - (lr * mh / (np.sqrt(vh) + eps_)).astype(np.float32))')
start, end = body.index('def auc_score'), body.index('# -----------------------------------------------------------------------------\n# 6.')
body = body[:start] + '''def auc_score(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')

''' + body[end:]
start = body.index('def run_mrt')
body = body[:start]  # replace MRT and privacy audit with corrected versions below
header = '''"""Completed Monte Carlo research prototype. All longitudinal outcomes are synthetic.
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

'''
(root / 'monte_carlo_digital_twin.py').write_text(header + body + (root / 'pipeline_tail.py').read_text(encoding='utf-8'), encoding='utf-8')
