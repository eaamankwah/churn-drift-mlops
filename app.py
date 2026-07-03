from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

import diagnostics
import scoring


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """Call the prediction function. Accepts JSON body with 'filepath' key."""
    filepath = request.get_json().get('filepath', None)
    if filepath:
        dataset = pd.read_csv(filepath)
    else:
        # Default to test data
        dataset = None
    preds = diagnostics.model_predictions(dataset)
    return jsonify(preds), 200


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """Return the F1 score of the deployed model."""
    f1 = scoring.score_model()
    return jsonify({'f1_score': f1}), 200


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """Return means, medians, and std devs for each numeric column."""
    stats = diagnostics.dataframe_summary()
    return jsonify(stats), 200


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diags():
    """Return timing, missing data, and dependency checks."""
    timing = diagnostics.execution_time()
    missing = diagnostics.dataframe_missing()
    packages = diagnostics.outdated_packages_list()
    return jsonify({
        'execution_time': {'ingestion_seconds': timing[0], 'training_seconds': timing[1]},
        'missing_data': missing,
        'outdated_packages': packages
    }), 200


if __name__ == "__main__":
    # Allow host/port/debug to be configured via environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ('1', 'true', 'yes')
    app.run(host=host, port=port, debug=debug, threaded=True)
