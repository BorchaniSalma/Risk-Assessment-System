"""
Risk Assessment System

Author : Salma Borchani

Date : 18th July 2024
"""
import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
model_path = Path(config['output_model_path'])


def confusion():
    """
    Calculate a confusion matrix using the test data and the deployed model, then save the confusion matrix as an image.

    Returns:
        None
    """
    try:
        # Load the test data
        testdata = pd.read_csv(test_data_path / 'testdata.csv')
        logging.info("Test data loaded successfully.")

        # Load the trained model
        with open(model_path / 'trainedmodel.pkl', 'rb') as file:
            model = pickle.load(file)
        logging.info("Trained model loaded successfully.")

        # Prepare the features and target variable
        X = testdata[['lastmonth_activity',
                      'lastyear_activity', 'number_of_employees']]
        y = testdata['exited']

        # Make predictions
        predicted = model.predict(X)
        logging.info("Model predictions made successfully.")

        # Calculate confusion matrix
        cf_matrix = metrics.confusion_matrix(y, predicted)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Save the plot
        output_path = model_path / 'confusionmatrix3.png'
        plt.savefig(output_path)
        logging.info(f"Confusion matrix saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    confusion()
