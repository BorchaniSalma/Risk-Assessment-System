import os
import json
import shutil

# Load config.json and correct path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']
model_path = os.path.join(config['output_model_path'])


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
    shutil.copy(
        os.getcwd() + '/' + output_folder_path + '/' + 'ingestedfiles.txt',
        prod_deployment_path)
    shutil.copy(
        model_path + '/' + 'trainedmodel.pkl',
        prod_deployment_path)
    shutil.copy(
        os.getcwd() + '/' + model_path + '/' + 'latestscore.txt',
        prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
