
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from deployment import store_model_into_pickle
from diagnostics import model_predictions, dataframe_summary, \
    missing_percentage, execution_time, outdated_packages_list
import json
import os
import pandas as pd
from reporting import confusion
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
test_data_path = os.path.join(config['test_data_path'])


# Check and read new data
# first, read ingestedfiles.txt
with open(os.path.join(os.getcwd(),
                       prod_deployment_path, 'ingestedfiles.txt'), 'r')\
        as ingested_data_file:
    data = (ingested_data_file.read())

# second, determine whether the source data folder has files that aren't
#  listed in ingestedfiles.txt
new_data = False
for filename in os.listdir(input_folder_path):
    # Deciding whether to proceed, part 1
    # if you found new data, you should proceed.
    if input_folder_path + "/" + filename not in data:
        new_data = True
# otherwise, do end the process here
if not new_data:
    print("New data was not ingested, add more to continue")
    exit(0)

if data:
    merge_multiple_dataframe()
    # Checking for model drift
    testdata = pd.read_csv(os.path.join(
        os.getcwd(), test_data_path, "testdata.csv"))
    score_model()

    with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") \
            as file:
        f1_old = float(file.read())

    with open(os.path.join(model_path, "latestscore.txt"), "r") as file:
        f1_new = float(file.read())
    # check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    if f1_old <= f1_new:
        print("No drift detected.")
        exit(0)
    else:
        print("Drift detected therefore retraining model")
        new_model = train_model()
        # Re-deployment
        # if you found evidence for model drift, re-run the deployment.py
        store_model_into_pickle()
        # Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model
        model_predictions()
        execution_time()
        dataframe_summary()
        missing_percentage()
        outdated_packages_list()
    confusion()
    os.system("python apicalls.py")
