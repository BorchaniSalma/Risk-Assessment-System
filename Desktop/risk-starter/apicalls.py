"""
Risk Assessment System

Author : Salma Borchani

Date : 18th July 2024
"""
import requests
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = Path(config['test_data_path'])
model_path = Path(config['output_model_path'])

# Define the API endpoints
prediction_endpoint = "http://127.0.0.1:5000/prediction"
scoring_endpoint = "http://127.0.0.1:5000/scoring"
summarystats_endpoint = "http://127.0.0.1:5000/summarystats"
diagnostics_endpoint = "http://127.0.0.1:5000/diagnostics"

# Function to make API requests and handle errors


def make_request(endpoint, method='get', data=None):
    try:
        if method == 'post':
            response = requests.post(endpoint, json=data)
        else:
            response = requests.get(endpoint)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request to {endpoint}: {e}")
        return None


# Make API requests
prediction_data = {'file_name': str(test_data_path / 'testdata.csv')}
response_pred = make_request(
    prediction_endpoint, method='post', data=prediction_data)

scoring_response = make_request(scoring_endpoint)
summarystats_response = make_request(summarystats_endpoint)
diagnostics_response = make_request(diagnostics_endpoint)

# Combine the responses into a dictionary
combined_responses = {
    "prediction": response_pred,
    "scoring": scoring_response,
    "summarystats": summarystats_response,
    "diagnostics": diagnostics_response,
}

# Write the combined responses to a file
output_file_path = model_path / 'apireturns3.txt'
with open(output_file_path, 'w') as score_file:
    json.dump(combined_responses, score_file)

logging.info(f"API responses written to {output_file_path}")
