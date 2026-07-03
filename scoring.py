"""
scoring.py  –  Score whichever model is currently deployed.

Reads model_metadata.json (written by training.py) to determine:
  - which feature set to apply (raw RAW_FEATURES for LR, ENGINEERED_FEATURES for RF)
  - whether to call engineer_features() before prediction

This makes scoring work correctly for BOTH the rubric-required LR model
and the improved RF model without any manual switch.
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import json

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path   = os.path.join(config['test_data_path'])
model_path       = os.path.join(config['output_model_path'])


def score_model():
    """
    Load the active deployed model, apply the correct feature pipeline,
    compute F1 on testdata.csv, and persist the score.
    Works for both LR (raw features) and RF (engineered features).
    """
    from training import engineer_features, load_metadata, RAW_FEATURES

    # ── Determine which features the active model expects ────────────────────
    meta = load_metadata()
    uses_engineering = meta.get('uses_engineering', False)
    feature_cols     = meta.get('feature_cols', RAW_FEATURES)
    active_model     = meta.get('active_model', 'logistic_regression')

    # ── Load and optionally transform test data ───────────────────────────────
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    if uses_engineering:
        test_df = engineer_features(test_df)

    X_test = test_df[feature_cols]
    y_test = pd.read_csv(
        os.path.join(test_data_path, 'testdata.csv')
    )['exited']

    # ── Load model and score ──────────────────────────────────────────────────
    model_file = os.path.join(model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    f1     = metrics.f1_score(y_test, y_pred, zero_division=0)
    print(f"[{active_model}]  F1 Score: {f1:.4f}")

    # ── Persist score ─────────────────────────────────────────────────────────
    os.makedirs(model_path, exist_ok=True)
    try:
        import dbsetup
        dbsetup.write_score_txt(model_path, f1)
    except Exception as e:
        score_file = os.path.join(model_path, 'latestscore.txt')
        with open(score_file, 'w') as f_out:
            f_out.write(str(f1))
        print(f"  Score file written (DB skipped: {e})")

    return f1


if __name__ == '__main__':
    score_model()
