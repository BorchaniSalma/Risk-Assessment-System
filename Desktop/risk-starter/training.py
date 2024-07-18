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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
default_model_path = Path(config['output_model_path'])


def train_model(model_output_path=default_model_path, dropped_columns=None):
    """
    Train a logistic regression model and save it to a file.

    This function trains a logistic regression model using data from 'finaldata.csv'
    and specified feature columns. It splits the data into training and testing sets,
    evaluates the model, and saves the trained model to the specified file path.

    Args:
        model_output_path (Path, optional): The path where the trained model will be saved.
        dropped_columns (list, optional): List of columns to drop from the dataset.

    Returns:
        None
    """
    if dropped_columns is None:
        dropped_columns = ['RowNumber', 'CustomerId', 'Surname', 'Exited']

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
    X = training_data.drop(columns=dropped_columns)
    y = training_data['Exited']
    logging.info("Starting training process.")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Train the model
    model = logit.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    logging.info("Training Metrics:")
    logging.info(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
    logging.info(f"F1 Score: {f1_score(y_train, y_pred_train)}")
    logging.info(f"Precision: {precision_score(y_train, y_pred_train)}")
    logging.info(f"Recall: {recall_score(y_train, y_pred_train)}")

    logging.info("Testing Metrics:")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred_test)}")
    logging.info(f"Precision: {precision_score(y_test, y_pred_test)}")
    logging.info(f"Recall: {recall_score(y_test, y_pred_test)}")

    # Save the trained model to a file
    model_output_path = model_output_path / 'bettertrainedmodel.pkl'
    with open(model_output_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    logging.info(f"Model saved to {model_output_path}")


if __name__ == '__main__':
    train_model()
