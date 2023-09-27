import pandas as pd
import pickle
import os
from sklearn import metrics
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


def score_model():
    """
    Calculate the F1 score of a trained model on test data and write the
    result to a file.

    This function loads the test data, a trained model, and calculates
    the F1 score
    for the model's predictions relative to the test data. It then writes
    the F1 score
    to the 'latestscore.txt' file in the model's output directory.

    Returns:
        None
    """
    testdata = pd.read_csv(test_data_path + '/' + 'testdata.csv')
    with open(model_path + '/' + 'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    X = testdata[['lastmonth_activity',
                  'lastyear_activity', 'number_of_employees']]
    y = testdata['exited']
    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)

    with open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'w') \
            as score_file:
        score_file.write(str(f1score))


if __name__ == '__main__':
    score_model()
