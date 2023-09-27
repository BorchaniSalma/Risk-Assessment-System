import requests
import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Define the API endpoints
# Replace with your actual host and port
prediction_endpoint = "http://127.0.0.1:5000/prediction"
# Replace with your actual host and port
scoring_endpoint = "http://127.0.0.1:5000/scoring"
# Replace with your actual host and port
summarystats_endpoint = "http://127.0.0.1:5000/summarystats"
# Replace with your actual host and port
diagnostics_endpoint = "http://127.0.0.1:5000/diagnostics"

# Make API requests
# Call the prediction endpoint with a sample file_name
response_pred = requests.post(
    'http://127.0.0.1:5000/prediction',
    json={
        'file_name': os.path.join(test_data_path + '/' + 'testdata.csv')}).text

# Call the scoring, summarystats, and diagnostics endpoints
scoring_response = requests.get(scoring_endpoint)
summarystats_response = requests.get(summarystats_endpoint)
diagnostics_response = requests.get(diagnostics_endpoint)

# Combine the responses into a dictionary
combined_responses = {
    "prediction": response_pred,
    "scoring": scoring_response.json(),
    "summarystats": summarystats_response.json(),
    "diagnostics": diagnostics_response.json(),
}
with open(os.path.join(os.getcwd(), model_path, 'apireturns.txt'), 'w') \
        as score_file:
    json.dump(combined_responses, score_file)
