import pandas as pd
import os
import json

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Function for data ingestion


def merge_multiple_dataframe():
    # Create an empty list to store DataFrames
    df_list = []

    # Check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.getcwd()+'/'+input_folder_path)
    for each_filename in filenames:
        file_path = (os.getcwd()+'/'+input_folder_path+'/'+each_filename)
        try:
            df1 = pd.read_csv(file_path)
            if not df1.empty:  # Check if the DataFrame is not empty
                df_list.append(df1)
            else:
                print(f"Skipping empty file: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file_path}")

    if df_list:
        result = pd.concat(df_list, ignore_index=True).drop_duplicates()
        output_file_path = (
            os.getcwd()+'/'+output_folder_path+'/'+'finaldata.csv')
        result.to_csv(output_file_path, index=False)
    else:
        print("No valid data found in the input files.")


if __name__ == '__main__':
    merge_multiple_dataframe()
