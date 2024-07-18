"""
Risk Assessment System

Author : Salma Borchani

Date : 18th July 2024
"""
import pandas as pd
import os
import json
import logging
from flask import Flask, jsonify, request
from pathlib import Path
from diagnostics import model_predictions, dataframe_summary, missing_percentage, execution_time, outdated_packages_list

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path'])
apireturns = config['apireturns']
test_data_path = Path(config['test_data_path'])

# Prediction Endpoint


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    try:
        file_name = request.get_json()['file_name']
        testdata = pd.read_csv(file_name)
        X_df = testdata.drop(['corporation', 'exited'], axis=1)
        predictions = model_predictions(X_df)
        return jsonify(predictions.tolist())
    except Exception as e:
        logging.error(f"Error in prediction endpoint: {e}")
        return jsonify({"error": str(e)}), 400

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scores():
    try:
        with open(model_path / 'latestscore.txt', 'r') as score_file:
            score = float(score_file.read())
            return jsonify({'score': score})
    except Exception as e:
        logging.error(f"Error in scoring endpoint: {e}")
        return jsonify({"error": str(e)}), 400

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    try:
        return jsonify(dataframe_summary())
    except Exception as e:
        logging.error(f"Error in summarystats endpoint: {e}")
        return jsonify({"error": str(e)}), 400

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    try:
        missing = missing_percentage()
        time = execution_time()
        outdated = outdated_packages_list()

        diagnostics = {
            'missing_percentage': missing,
            'execution_time': time,
            'outdated_packages': outdated
        }

        return jsonify(diagnostics)
    except Exception as e:
        logging.error(f"Error in diagnostics endpoint: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/apireturns", methods=['GET'])
def name():
    return str(model_path / apireturns)


if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True, threaded=True)
