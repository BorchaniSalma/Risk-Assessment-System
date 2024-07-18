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


def clean_test_dataset(df_list):
    """
    Concatenate a list of DataFrames, remove duplicate rows, and convert strings to numbers.

    This function takes a list of pandas DataFrames, concatenates them
    into a single DataFrame, removes any duplicate rows, and converts
    string values to numerical values using dictionaries.

    Args:
        df_list (list): A list of pandas DataFrames to be concatenated and cleaned.

    Returns:
        pd.DataFrame: A concatenated DataFrame with duplicate rows removed
                      and strings converted to numerical values.
    """
    # Ensure df_list is a list
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    result = pd.concat(df_list, ignore_index=True).drop_duplicates()

    # Convert strings to numbers
    for column in result.select_dtypes(include=['object']).columns:
        unique_values = result[column].unique()
        value_to_number = {val: num for num, val in enumerate(unique_values)}
        result[column] = result[column].map(value_to_number)

    return result


def score_model(dropped_columns=None):
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
    test_data = clean_test_dataset(test_data)

    if dropped_columns is None:
        dropped_columns = ['RowNumber', 'CustomerId', 'Surname', 'Exited']

    # Ensure target column 'Exited' is present
    if 'Exited' not in test_data.columns:
        raise ValueError("Target column 'Exited' not found in the test data")

    # Load the trained model
    with open(model_path / 'bettertrainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Define features and target variable
    X = test_data.drop(columns=dropped_columns)
    y = test_data['Exited']

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
