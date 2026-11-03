import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline

# Define the paths and features
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/FINAL_dataset_with_fitbit.csv"

# These are the wearable features used in your simulation
WEARABLE_COLS = ["nremhr", "rmssd", "spo2", "sleep_hours", "stress_score", "physical_activity"]
# These are typical clinical features (adjust if your baseline uses different ones)
CLINICAL_COLS = ["age", "gender_male", "bmi", "smoking_history_never", "hypertension"] 

def run_diagnostics():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    target = "heart_disease"
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found.")
        return

    y = df[target]
    
    # 1. Check Missingness
    print("\n--- 1. WEARABLE DATA MISSINGNESS ---")
    missing_pct = df[WEARABLE_COLS].isna().mean() * 100
    for col, pct in missing_pct.items():
        print(f"{col:20}: {pct:.1f}% missing")

    # Filter for complete cases (rows where NO wearable data is missing)
    complete_mask = df[WEARABLE_COLS].notna().all(axis=1)
    df_complete = df[complete_mask]
    y_complete = df_complete[target]
    
    print(f"\nTotal patients: {len(df)}")
    print(f"Patients with COMPLETE wearable data: {len(df_complete)} ({len(df_complete)/len(df)*100:.1f}%)")

    if len(df_complete) < 100:
        print("Not enough complete cases to run reliable models. Exiting.")
        return

    # Split the complete cases
    X_clin = df_complete[CLINICAL_COLS]
    X_wear = df_complete[WEARABLE_COLS]
    X_comb = df_complete[CLINICAL_COLS + WEARABLE_COLS]
    
    X_train_c, X_test_c, y_train, y_test = train_test_split(X_clin, y_complete, test_size=0.3, random_state=42)
    X_train_w, X_test_w, _, _ = train_test_split(X_wear, y_complete, test_size=0.3, random_state=42)
    X_train_cb, X_test_cb, _, _ = train_test_split(X_comb, y_complete, test_size=0.3, random_state=42)

    # Define a simple model (Logistic Regression is good for quick signal checking)
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=500))

    print("\n--- 2. MODEL EVALUATION (ON COMPLETE CASES ONLY) ---")
    
    # Model A: Wearable Only
    m_wear = model().fit(X_train_w, y_train)
    prob_wear = m_wear.predict_proba(X_test_w)[:, 1]
    auc_wear = roc_auc_score(y_test, prob_wear)
    pr_auc_wear = average_precision_score(y_test, prob_wear)
    
    # Model B: Clinical Only
    m_clin = model().fit(X_train_c, y_train)
    prob_clin = m_clin.predict_proba(X_test_c)[:, 1]
    auc_clin = roc_auc_score(y_test, prob_clin)
    pr_auc_clin = average_precision_score(y_test, prob_clin)

    # Model C: Combined (Clinical + Wearable)
    m_comb = model().fit(X_train_cb, y_train)
    prob_comb = m_comb.predict_proba(X_test_cb)[:, 1]
    auc_comb = roc_auc_score(y_test, prob_comb)
    pr_auc_comb = average_precision_score(y_test, prob_comb)

    print(f"{'Model':<20} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-" * 45)
    print(f"{'Wearable Only':<20} | {auc_wear:<10.3f} | {pr_auc_wear:<10.3f}")
    print(f"{'Clinical Only':<20} | {auc_clin:<10.3f} | {pr_auc_clin:<10.3f}")
    print(f"{'Combined':<20} | {auc_comb:<10.3f} | {pr_auc_comb:<10.3f}")
    
    delta_auc = auc_comb - auc_clin
    print(f"\nIncremental Value (Delta AUC): {delta_auc:+.3f}")
    
    if auc_wear < 0.55:
        print("\nCONCLUSION: The wearable data has near-random predictive signal in isolation.")
    if delta_auc < 0.02:
        print("CONCLUSION: Adding wearable data provides negligible improvement over clinical data alone.")

if __name__ == "__main__":
    run_diagnostics()