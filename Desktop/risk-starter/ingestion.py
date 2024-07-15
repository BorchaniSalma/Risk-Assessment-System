'''
Risk Assessment System

Author : Salma Borchani

Date : 15th July 2024
'''
from pathlib import Path
import pandas as pd
import json
import logging
import time
from glob import glob

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

INPUT_FOLDER_PATH = Path(config['input_folder_path'])
OUTPUT_FOLDER_PATH = Path(config['output_folder_path'])
OUTPUT_CSV_FILE = OUTPUT_FOLDER_PATH / 'finaldata.csv'
INGESTED_FILES_LOG = OUTPUT_FOLDER_PATH / \
    f"ingestedfiles_{time.strftime('%y%m%d%H%M%S')}.txt"


def clean_dataset(df_list):
    """
    Concatenate a list of DataFrames and remove duplicate rows.

    This function takes a list of pandas DataFrames, concatenates them
    into a single DataFrame, and then removes any duplicate rows.

    Args:
        df_list (list): A list of pandas DataFrames to be concatenated and cleaned.

    Returns:
        pd.DataFrame: A concatenated DataFrame with duplicate rows removed.
    """
    result = pd.concat(df_list, ignore_index=True).drop_duplicates()
    return result


def merge_multiple_dataframe():
    """
    Merge multiple CSV files into a single CSV file and record ingested files.

    This function reads CSV files from the specified input folder,
     compiles them
    together into a single DataFrame, removes duplicate rows, and writes the
    resulting DataFrame to a CSV file in the specified output folder. It also
    records the names of the ingested files in 'ingestedfiles.txt'
    in the output folder.

    Returns:
        None
    """
    df_list = []  # Create an empty list to store DataFrames
    ingested_files = []   # List to store the names of ingested files

    # Use glob to find all CSV files in the input folder
    csv_files = glob(str(INPUT_FOLDER_PATH / '*.csv'))

    for each_filename in csv_files:
        try:
            df = pd.read_csv(each_filename)
            if not df.empty:  # Check if the DataFrame is not empty
                df_list.append(df)
                # Record the ingested file
                ingested_files.append(Path(each_filename).name)
            else:
                logging.warning(f"Skipping empty file: {each_filename}")
        except pd.errors.EmptyDataError:
            logging.warning(f"Skipping empty file: {each_filename}")
        except Exception as e:
            logging.error(f"Error reading {each_filename}: {e}")

    if df_list:
        result = clean_dataset(df_list)
        result.to_csv(OUTPUT_CSV_FILE, index=False)
        logging.info(f"Merged data saved to {OUTPUT_CSV_FILE}")
    else:
        logging.warning("No valid data found in the input files.")

    # Save the list of ingested files to a timestamped ingestedfiles log
    with INGESTED_FILES_LOG.open('w') as file:
        for ingested_file in ingested_files:
            file.write(ingested_file + '\n')
    logging.info(f"Ingested files list saved to {INGESTED_FILES_LOG}")


if __name__ == '__main__':
    merge_multiple_dataframe()
