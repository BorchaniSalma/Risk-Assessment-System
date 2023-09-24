import pandas as pd
from datetime import datetime
import json
import os

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']

filename = 'finaldata.csv'
outputlocation = (
    os.getcwd()+'/'+output_folder_path+'/'+'ingestedfiles.txt')
data = pd.read_csv(output_folder_path+'/'+filename)
dateTimeObj = datetime.now()
thetimenow = str(dateTimeObj.year) + '/' + \
    str(dateTimeObj.month) + '/'+str(dateTimeObj.day)
allrecords = [output_folder_path, filename, len(data.index), thetimenow]
MyFile = open(outputlocation, 'w')
for element in allrecords:
    MyFile.write(str(element))
