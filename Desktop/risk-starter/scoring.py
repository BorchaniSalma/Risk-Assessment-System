"""
Risk Assessment System

Author : Salma Borchani

Date : 15th July 2024
"""
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from sklearn import metrics

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
model_path = Path(config['prod_deployment_path'])


def score_model():
    """
    Calculate the F1 score of a trained model on test data and write the result to a file.

    This function loads the test data, a trained model, and calculates the F1 score
    for the model's predictions relative to the test data. It then writes the F1 score
    to the 'latestscore.txt' file in the model's output directory.

    Returns:
        None
    """
    # Load the test data
    test_data = pd.read_csv(test_data_path / 'testdata.csv')

    # Load the trained model
    with open(model_path / 'bettertrainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Define features and target variable
    X = test_data[['lastmonth_activity',
                   'lastyear_activity', 'number_of_employees']]
    y = test_data['exited']

    # Make predictions
    predictions = model.predict(X)

    # Calculate F1 score
    f1_score = metrics.f1_score(y, predictions)
    logging.info(f"Calculated F1 score: {f1_score}")

    # Write the F1 score to a file
    score_file_path = model_path / 'latestscore.txt'
    with open(score_file_path, 'w') as score_file:
        score_file.write(str(f1_score))
    logging.info(f"F1 score written to {score_file_path}")


if __name__ == '__main__':
    score_model()
