import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for reporting
def confusion():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    testdata = pd.read_csv(test_data_path + '/' + 'testdata.csv')
    with open(model_path + '/' + 'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    X = testdata[['lastmonth_activity',
                  'lastyear_activity', 'number_of_employees']]
    y = testdata['exited']
    predicted = model.predict(X)
    cf_matrix = metrics.confusion_matrix(y, predicted)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(model_path + '/' + 'confusionmatrix.png')


if __name__ == '__main__':
    confusion()
