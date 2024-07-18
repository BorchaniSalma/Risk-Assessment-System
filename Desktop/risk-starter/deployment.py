
"""
Risk Assessment System

Author : Salma Borchani

Date : 15th July 2024
"""

import json
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = Path(config['output_folder_path'])
prod_deployment_path = Path(config['prod_deployment_path'])
output_folder_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path'])


def store_model_into_pickle():
    """
    Copy necessary files to the deployment directory.

    This function copies the latest 'trainedmodel.pkl' file,
    'latestscore.txt' value,
    and 'ingestedfiles.txt' file from their respective locations to
    the deployment directory specified in 'config.json'.

    Returns:
        None
    """
    files_to_copy = {
        'ingestedfiles.txt': output_folder_path / 'ingestedfiles.txt',
        'bettertrainedmodel.pkl': model_path / 'bettertrainedmodel.pkl',
        'latestscore.txt': model_path / 'latestscore.txt'
    }

    for filename, filepath in files_to_copy.items():
        try:
            destination = prod_deployment_path / filename
            shutil.copy(filepath, destination)
            logging.info(f"Copied {filename} to {destination}")
        except FileNotFoundError:
            logging.error(f"File {filename} not found at {filepath}")
        except Exception as e:
            logging.error(f"Error copying {filename} to {destination}: {e}")


if __name__ == '__main__':
    store_model_into_pickle()
