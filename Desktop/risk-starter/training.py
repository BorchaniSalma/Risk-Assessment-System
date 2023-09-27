import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


def train_model():
    """
    Train a logistic regression model and save it to a file.

    This function trains a logistic regression model using data
    from 'finaldata.csv'
    and the columns 'lastmonth_activity', 'lastyear_activity',
    and 'number_of_employees'
    as features, and 'exited' as the target variable.
    The trained model is saved as
    'trainedmodel.pkl' in the model's output directory specified in
    'config.json'.

    Returns:
        None
    """
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False,
                               fit_intercept=True, intercept_scaling=1,
                               l1_ratio=None,
                               max_iter=100, multi_class='auto', n_jobs=None,
                               penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001,
                               verbose=0,
                               warm_start=False)

    trainingdata = pd.read_csv(dataset_csv_path + '/' + 'finaldata.csv')
    X = trainingdata[['lastmonth_activity',
                      'lastyear_activity', 'number_of_employees']]
    y = trainingdata['exited']
    model = logit.fit(X, y)

    # Save the trained model to a file
    pickle.dump(model, open(model_path + '/' 'trainedmodel.pkl', 'wb'))


if __name__ == '__main__':
    train_model()
