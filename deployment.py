"""
deployment.py  –  Promote trained model artefacts to production_deployment/.
Now also copies model_metadata.json so the production environment knows
which algorithm is active (LR vs RF) and what features to use.
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path    = os.path.join(config['output_folder_path'])
model_path          = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def store_model_into_pickle(model=None):
    """Copy active model artefacts and metadata to production_deployment/."""
    os.makedirs(prod_deployment_path, exist_ok=True)

    # Core artefacts (always present)
    for filename in ['trainedmodel.pkl', 'latestscore.txt']:
        src = os.path.join(model_path, filename)
        dst = os.path.join(prod_deployment_path, filename)
        shutil.copy2(src, dst)
        print(f"Copied: {filename}")

    # Feature list and metadata (may not exist for very first LR-only run)
    for filename in ['feature_columns.pkl', 'model_metadata.json',
                     'trainedmodel_lr.pkl', 'trainedmodel_rf.pkl']:
        src = os.path.join(model_path, filename)
        if os.path.exists(src):
            dst = os.path.join(prod_deployment_path, filename)
            shutil.copy2(src, dst)
            print(f"Copied: {filename}")

    # Ingested files audit trail
    src = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    dst = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    shutil.copy2(src, dst)
    print(f"Copied: ingestedfiles.txt")

    print("Deployment complete.")


if __name__ == '__main__':
    store_model_into_pickle()
