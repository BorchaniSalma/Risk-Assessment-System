
import subprocess
import pandas as pd
import timeit
import os
import json
import pickle


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])
# Function to get model predictions


def model_predictions(X_dataframe):

    # read the deployed model and a test dataset, calculate predictions
    with open(model_path+'/'+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    X = X_dataframe[['lastmonth_activity', 'lastyear_activity',
                     'number_of_employees']]
    # return value should be a list containing all predictions
    y_predicted = model.predict(X)
    return y_predicted

# Function to get summary statistics


def dataframe_summary():

    # calculate summary statistics here
    data_df = pd.read_csv(dataset_csv_path+'/'+'finaldata.csv')
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
    Calculates percentage of missing data for each column
    in finaldata.csv

    Returns:
        list[dict]: Each dict contains column name and percentage
    """
    data_df = pd.read_csv(dataset_csv_path+'/'+'finaldata.csv')
    missing_list = {col: {'percentage': perc} for col, perc in zip(
        data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)}

    return missing_list
# Function to get timings


def execution_time():
    # calculate timing of training.py and ingestion.py
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
    Function execute command and returns stdout as a list

    input: command as a list ex ['pip','list','--outdated']
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
# Function to check dependencies


def outdated_packages_list():
    pip_outdated = execute_command(["pip", "list", "--outdated"])

    # read current requirements file
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
    testdata = pd.read_csv(test_data_path+'/'+'testdata.csv')
    X_df = testdata.drop(['corporation', 'exited'], axis=1)
    print(model_predictions(X_df))
    print(dataframe_summary())
    print(execution_time())
    print(outdated_packages_list())
