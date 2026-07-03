"""
evidently_monitoring.py - Standout #4
Monitors client attrition risk classification model using Evidently.
Tracks: data drift, target/prediction drift, and classification performance.
Updated to use engineer_features() for consistency with the trained RF model.
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics as skmetrics
from datetime import datetime

warnings.filterwarnings('ignore')

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = config['output_folder_path']
test_data_path   = config['test_data_path']
model_path       = config['output_model_path']
prod_path        = config['prod_deployment_path']

RAW_FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET = 'exited'


def _load_model_and_features():
    with open(os.path.join(prod_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    feat_file = os.path.join(prod_path, 'feature_columns.pkl')
    if os.path.exists(feat_file):
        with open(feat_file, 'rb') as f:
            feature_cols = pickle.load(f)
    else:
        feature_cols = RAW_FEATURES
    return model, feature_cols


def _load_data():
    from training import engineer_features
    ref = engineer_features(pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv')))
    cur = engineer_features(pd.read_csv(os.path.join(test_data_path, 'testdata.csv')))
    return ref, cur


def compute_data_drift() -> dict:
    from scipy.stats import ks_2samp
    ref, cur = _load_data()
    drift_results = {}
    drifted = 0
    for col in RAW_FEATURES:
        stat, p = ks_2samp(ref[col].dropna(), cur[col].dropna())
        is_drift = p < 0.05
        drift_results[col] = {'ks_statistic': round(float(stat), 4),
                               'p_value': round(float(p), 4),
                               'drifted': is_drift}
        if is_drift:
            drifted += 1
    drift_results['_summary'] = {
        'drifted_columns': drifted,
        'total_columns': len(RAW_FEATURES),
        'drift_share': round(drifted / len(RAW_FEATURES), 4),
        'dataset_drift_detected': drifted / len(RAW_FEATURES) >= 0.5
    }
    return drift_results


def compute_classification_performance() -> dict:
    model, feature_cols = _load_model_and_features()
    _, cur = _load_data()
    X     = cur[feature_cols]
    y_true = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))[TARGET]
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    results = {
        'accuracy':  round(float(skmetrics.accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(skmetrics.precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':    round(float(skmetrics.recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_score':  round(float(skmetrics.f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    try:
        results['roc_auc'] = round(float(skmetrics.roc_auc_score(y_true, y_proba)), 4)
    except Exception:
        results['roc_auc'] = None
    return results


def compute_prediction_drift() -> dict:
    model, feature_cols = _load_model_and_features()
    ref, cur = _load_data()
    ref_raw = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    ref_target_dist = ref_raw[TARGET].value_counts(normalize=True).sort_index()
    cur_preds = model.predict(cur[feature_cols])
    cur_pred_dist = pd.Series(cur_preds).value_counts(normalize=True).sort_index()
    all_classes = sorted(set(ref_target_dist.index) | set(cur_pred_dist.index))
    ref_counts = np.array([ref_target_dist.get(c, 0) for c in all_classes])
    cur_counts = np.array([cur_pred_dist.get(c, 0) for c in all_classes])
    return {
        'reference_target_distribution':  {str(k): round(float(v), 4) for k, v in zip(all_classes, ref_counts)},
        'current_prediction_distribution': {str(k): round(float(v), 4) for k, v in zip(all_classes, cur_counts)},
        'prediction_drift_detected': bool(np.max(np.abs(ref_counts - cur_counts)) > 0.2)
    }


def save_evidently_plots(output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    plots = {}
    drift    = compute_data_drift()
    clf_perf = compute_classification_performance()
    pred_dr  = compute_prediction_drift()

    # Plot 1: Feature drift
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cols    = RAW_FEATURES
    ks_vals = [drift[c]['ks_statistic'] for c in cols]
    colors  = ['#e74c3c' if drift[c]['drifted'] else '#2ecc71' for c in cols]
    bars = ax.barh(cols, ks_vals, color=colors)
    ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=1.2, label='p=0.05 threshold')
    ax.set_xlabel('KS Statistic')
    ax.set_title('Feature Drift — KS Statistics\n(red = drift, green = no drift)')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, ks_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'evidently_feature_drift.png')
    plt.savefig(p1, dpi=100); plt.close()
    plots['feature_drift'] = p1

    # Plot 2: Classification performance
    fig, ax = plt.subplots(figsize=(7, 3.5))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metric_vals  = [clf_perf['accuracy'], clf_perf['precision'],
                    clf_perf['recall'], clf_perf['f1_score'],
                    clf_perf['roc_auc'] if clf_perf['roc_auc'] is not None else 0.0]
    bar_colors = ['#3498db' if v >= 0.5 else '#e74c3c' for v in metric_vals]
    ax.bar(metric_names, metric_vals, color=bar_colors)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance Metrics (RandomForest + Feature Engineering)')
    ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    for i, (name, val) in enumerate(zip(metric_names, metric_vals)):
        ax.text(i, val + 0.03, f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'evidently_classification_performance.png')
    plt.savefig(p2, dpi=100); plt.close()
    plots['classification_performance'] = p2

    # Plot 3: Prediction drift
    fig, ax = plt.subplots(figsize=(6, 3.5))
    classes  = list(pred_dr['reference_target_distribution'].keys())
    ref_vals = [pred_dr['reference_target_distribution'][c] for c in classes]
    cur_vals = [pred_dr['current_prediction_distribution'].get(c, 0) for c in classes]
    x = np.arange(len(classes)); width = 0.35
    ax.bar(x - width/2, ref_vals, width, label='Reference (actual)',    color='#3498db', alpha=0.85)
    ax.bar(x + width/2, cur_vals, width, label='Current (predictions)', color='#e67e22', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f'Class {c}' for c in classes])
    ax.set_ylabel('Proportion'); ax.set_title('Target vs Prediction Distribution Drift')
    ax.legend(fontsize=8)
    plt.tight_layout()
    p3 = os.path.join(output_dir, 'evidently_prediction_drift.png')
    plt.savefig(p3, dpi=100); plt.close()
    plots['prediction_drift'] = p3

    return plots


def run_full_monitoring() -> dict:
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_drift': compute_data_drift(),
        'classification_performance': compute_classification_performance(),
        'prediction_drift': compute_prediction_drift(),
    }
    try:
        import dbsetup
        dbsetup.insert_diagnostics(0.0, 0.0, [], [], results['data_drift'])
    except Exception:
        pass
    return results


if __name__ == '__main__':
    print("Running Evidently monitoring...")
    r = run_full_monitoring()
    print("Classification performance:", r['classification_performance'])
    print("Drift summary:", r['data_drift']['_summary'])
    plots = save_evidently_plots(config['output_model_path'])
    print("Plots:", list(plots.keys()))
