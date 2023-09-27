import subprocess
import pandas as pd
import os
import json
from flask import Flask, jsonify, request
from diagnostics import model_predictions, dataframe_summary, \
    missing_percentage, execution_time, outdated_packages_list
from scoring import score_model

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None
model_path = os.path.join(config['output_model_path'])
apireturns = config['apireturns']
test_data_path = os.path.join(config['test_data_path'])


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    file_name = request.get_json()['file_name']
    testdata = pd.read_csv(file_name)
    X_df = testdata.drop(['corporation', 'exited'], axis=1)
    predictions = model_predictions(X_df)
    return jsonify(predictions.tolist())

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scores():
    with open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'r') \
            as score_file:
        score = float(score_file.read())
        return jsonify({'score': score})
# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    return jsonify(dataframe_summary())


# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    # check timing and percent NA values
    missing = missing_percentage()
    time = execution_time()
    outdated = outdated_packages_list()

    diagnostics = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return jsonify(diagnostics)


@app.route("/apireturns", methods=['GET'])
def name():
    return model_path + '/' + apireturns


if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True, threaded=True)
