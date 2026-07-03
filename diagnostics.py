"""
diagnostics.py  –  Runtime diagnostics for the deployed model.
Uses load_metadata() from training.py to apply the correct feature
pipeline for whichever model (LR or RF) is currently active.
"""
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path   = os.path.join(config['test_data_path'])
prod_path        = os.path.join(config['prod_deployment_path'])


def model_predictions(dataset=None):
    """Run the deployed model; applies feature engineering only when RF is active."""
    from training import engineer_features, load_metadata, RAW_FEATURES

    meta             = load_metadata()
    uses_engineering = meta.get('uses_engineering', False)
    feature_cols     = meta.get('feature_cols', RAW_FEATURES)

    model_file = os.path.join(prod_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    if dataset is None:
        dataset = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    if uses_engineering:
        dataset = engineer_features(dataset)

    X = dataset[feature_cols]
    return model.predict(X).tolist()


def dataframe_summary():
    df   = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        {'column': c, 'mean': df[c].mean(), 'median': df[c].median(), 'std': df[c].std()}
        for c in cols
    ]


def dataframe_missing():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    return [
        {'column': c, 'missing_pct': df[c].isna().mean() * 100}
        for c in df.columns
    ]


def execution_time():
    t0 = timeit.default_timer()
    os.system(f'{sys.executable} ingestion.py')
    ingestion_time = timeit.default_timer() - t0

    t0 = timeit.default_timer()
    os.system(f'{sys.executable} training.py')
    training_time = timeit.default_timer() - t0

    return [ingestion_time, training_time]


def outdated_packages_list():
    with open('requirements.txt', 'r') as f:
        requirements = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format=json'],
        capture_output=True, text=True
    )
    installed = {}
    if result.returncode == 0:
        import json as _j
        for pkg in _j.loads(result.stdout):
            installed[pkg['name'].lower()] = pkg['version']

    outdated_info = []
    for req in requirements:
        pkg_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip()
        current  = installed.get(pkg_name.lower(), 'N/A')
        latest   = 'N/A'
        try:
            r = subprocess.run(
                [sys.executable, '-m', 'pip', 'index', 'versions', pkg_name],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if 'LATEST:' in line:
                        latest = line.split('LATEST:')[-1].strip(); break
                    if 'Available versions:' in line:
                        latest = line.split('Available versions:')[-1].strip().split(',')[0].strip(); break
        except Exception:
            pass
        outdated_info.append({'package': pkg_name, 'current_version': current, 'latest_version': latest})

    return outdated_info


def save_diagnostics_snapshot(ingestion_time=0.0, training_time=0.0):
    os.makedirs('olddiagnostics', exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    stats   = dataframe_summary()
    missing = dataframe_missing()

    snapshot = {
        'timestamp':      ts,
        'ingestion_time': ingestion_time,
        'training_time':  training_time,
        'summary_stats':  stats,
        'missing_data':   missing,
    }
    snap_path = os.path.join('olddiagnostics', f'diagnostics_{ts}.json')
    with open(snap_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    try:
        import dbsetup
        dbsetup.insert_diagnostics(ingestion_time, training_time, stats, missing)
    except Exception as e:
        print(f"[diagnostics] DB write skipped: {e}")

    print(f"Diagnostics snapshot: {snap_path}")
    return snap_path


if __name__ == '__main__':
    print("=== Model Predictions ===")
    print(model_predictions())
    print("\n=== Summary Statistics ===")
    for s in dataframe_summary():
        print(s)
    print("\n=== Missing Data ===")
    for m in dataframe_missing():
        print(m)
