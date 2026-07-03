"""
fullprocess.py – Full process automation.
Checks for new data → checks model drift → retrains/redeploys if needed.
Uses train_best_model() so BOTH LR and RF are evaluated on every retrain
cycle and the winner is deployed automatically.
"""
import os
import json
import pickle
import pandas as pd
from sklearn import metrics

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import evidently_monitoring

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path    = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path   = config['output_folder_path']

# ── Check for new source files ────────────────────────────────────────────────
ingested_record     = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
previously_ingested = []
if os.path.exists(ingested_record):
    with open(ingested_record, 'r') as f:
        previously_ingested = [l.strip() for l in f if l.strip()]

try:
    import dbsetup
    db_ingested = dbsetup.get_ingested_files()
    previously_ingested = list(set(previously_ingested + db_ingested))
except Exception:
    pass

source_files = [
    os.path.join(input_folder_path, f)
    for f in os.listdir(input_folder_path) if f.endswith('.csv')
]
new_files = [f for f in source_files if f not in previously_ingested]

# ── Checkpoint 1: exit if no new data ────────────────────────────────────────
if not new_files:
    print("No new data found. Exiting.")
    exit(0)

print(f"New data: {new_files}. Running ingestion...")
ingestion.merge_multiple_dataframe()

# ── Check for model drift ─────────────────────────────────────────────────────
with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
    deployed_score = float(f.read().strip())
print(f"Deployed F1: {deployed_score:.4f}")

# Score deployed model on newly ingested data using its own feature pipeline
meta             = training.load_metadata()
uses_engineering = meta.get('uses_engineering', False)
feature_cols     = meta.get('feature_cols', training.RAW_FEATURES)

new_data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
if uses_engineering:
    new_data_e = training.engineer_features(new_data)
    X_new = new_data_e[feature_cols]
else:
    X_new = new_data[feature_cols]
y_new = new_data['exited']

with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
    deployed_model = pickle.load(f)

new_score   = metrics.f1_score(y_new, deployed_model.predict(X_new), zero_division=0)
model_drift = new_score < deployed_score
print(f"New data score: {new_score:.4f}  |  Drift: {model_drift}")

# ── Checkpoint 2: exit if no drift ───────────────────────────────────────────
if not model_drift:
    print(f"No drift (new={new_score:.4f} >= deployed={deployed_score:.4f}). Exiting.")
    exit(0)

print("Drift detected — running train_best_model() (LR vs RF comparison)...")

# ── Retrain: evaluate both LR and RF, deploy winner ──────────────────────────
training.train_best_model()
scoring.score_model()
deployment.store_model_into_pickle()

# ── Diagnostics + time trends ─────────────────────────────────────────────────
import timeit, sys
t0    = timeit.default_timer(); os.system(f'{sys.executable} ingestion.py'); ing_t = timeit.default_timer() - t0
t0    = timeit.default_timer(); os.system(f'{sys.executable} training.py');  trn_t = timeit.default_timer() - t0
diagnostics.save_diagnostics_snapshot(ing_t, trn_t)

# ── Evidently monitoring ──────────────────────────────────────────────────────
print("Running Evidently monitoring...")
ev_results = evidently_monitoring.run_full_monitoring()
ev_plots   = evidently_monitoring.save_evidently_plots(config['output_model_path'])
print("Evidently plots:", list(ev_plots.keys()))

# ── Reporting ─────────────────────────────────────────────────────────────────
reporting.score_model()
reporting.generate_pdf_report()

# ── API calls (Flask server must be running separately) ───────────────────────
try:
    import apicalls  # noqa: F401
    print("API calls complete.")
except Exception as e:
    print(f"API calls skipped (start Flask server first): {e}")

print("Full process complete.")
