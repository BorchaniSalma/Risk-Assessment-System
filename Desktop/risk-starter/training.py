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
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path'])


def train_model():
    """
    Train a logistic regression model and save it to a file.

    This function trains a logistic regression model using data
    from 'finaldata.csv'
    and the columns 'lastmonth_activity', 'lastyear_activity',
    and 'number_of_employees'
    as features, and 'exited' as the target variable.
    The trained model is saved as
    'bettertrainedmodel.pkl' in the model's output directory specified in
    'config.json'.

    Returns:
        None
    """
    # Define logistic regression model with appropriate parameters
    logit = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        n_jobs=None, penalty='l2', random_state=0, solver='liblinear',
        tol=0.0001, verbose=0, warm_start=False
    )

    # Load the training data
    training_data_path = dataset_csv_path / 'finaldata.csv'
    training_data = pd.read_csv(training_data_path)
    X = training_data[['lastmonth_activity',
                       'lastyear_activity', 'number_of_employees']]
    y = training_data['exited']
    logging.info("Starting training process.")

    # Train the model
    model = logit.fit(X, y)
    logging.info("Model training completed.")

    # Save the trained model to a file
    model_output_path = model_path / 'bettertrainedmodel.pkl'
    with open(model_output_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    logging.info(f"Model saved to {model_output_path}")


if __name__ == '__main__':
    train_model()
