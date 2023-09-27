import subprocess
import timeit
import os
import json
import pickle
import pandas as pd

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])


def model_predictions(X_dataframe):
    """
    Get model predictions for the given input data.

    Args:
        X_dataframe (pd.DataFrame): Input data frame with columns
        'lastmonth_activity','lastyear_activity', and 'number_of_employees'.

    Returns:
        list: List containing model predictions.
    """
    with open(model_path + '/' + 'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    X = X_dataframe[['lastmonth_activity',
                     'lastyear_activity', 'number_of_employees']]
    y_predicted = model.predict(X)
    return y_predicted


def dataframe_summary():
    """
    Calculate summary statistics for the dataset.

    Returns:
        dict: Dictionary containing summary statistics for numerical columns.
    """
    data_df = pd.read_csv(dataset_csv_path + '/' + 'finaldata.csv')
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')

    statistics_dict = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()
        statistics_dict[col] = {'mean': mean, 'median': median, 'std': std}

    return statistics_dict


def missing_percentage():
    """
    Calculate the percentage of missing data for each column in finaldata.csv.

    Returns:
        list[dict]: List of dictionaries, where each dictionary contains
         column name and percentage of missing data.
    """
    data_df = pd.read_csv(dataset_csv_path + '/' + 'finaldata.csv')
    missing_list = {col: {'percentage': perc} for col, perc in zip(
        data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)}

    return missing_list


def execution_time():
    """
    Calculate the execution time of ingestion.py and training.py.

    Returns:
        dict: Dictionary containing the execution time
        of ingestion and training.
    """
    time_dict = {}
    starttime_ingestion = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing_ingestion = timeit.default_timer() - starttime_ingestion
    starttime_training = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing_training = timeit.default_timer() - starttime_training
    time_dict = {'ingestion_time': timing_ingestion,
                 'training_time': timing_training}
    return time_dict


def execute_command(cmd_list):
    """
    Execute a command and return its stdout as a list.

    Args:
        cmd_list (list): List containing the command and its arguments.

    Returns:
        list: List of lines from the command's stdout.
    """
    process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
    results = []
    while True:
        output = process.stdout.readline()
        if output.decode('utf8') == '' and process.poll() is not None:
            break
        if output:
            results.append(output.decode('utf8').strip().split())
    return results


def outdated_packages_list():
    """
    Get a list of outdated Python packages based on the requirements file.

    Returns:
        list: List of outdated package information.
    """
    pip_outdated = execute_command(["pip", "list", "--outdated"])

    with open("requirements.txt", 'rb') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]

    results = []
    for line in lines:
        x = line.split("==")
        for y in pip_outdated:
            if x[0] == y[0]:
                results.append(f"{y[0]} - {y[1]} - {y[2]}")

    return results


if __name__ == '__main__':
    testdata = pd.read_csv(test_data_path + '/' + 'testdata.csv')
    X_df = testdata.drop(['corporation', 'exited'], axis=1)
    print(model_predictions(X_df))
    print(dataframe_summary())
    print(missing_percentage())
    print(execution_time())
    print(outdated_packages_list())
