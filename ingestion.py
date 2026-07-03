import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    """Check for datasets, compile them together, and write to an output file."""
    os.makedirs(output_folder_path, exist_ok=True)

    # Detect all csv files in the input directory
    all_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    file_paths = [os.path.join(input_folder_path, f) for f in all_files]

    # Read and combine all csv files into a single DataFrame
    dataframes = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # De-duplicate
    combined_df.drop_duplicates(inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Write the compiled dataset
    output_csv = os.path.join(output_folder_path, 'finaldata.csv')
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved combined dataset to {output_csv} ({len(combined_df)} rows)")

    # Save ingestion record to .txt AND SQL database (Standout #3)
    try:
        import dbsetup
        dbsetup.write_ingested_txt(output_folder_path, file_paths)
    except Exception as e:
        # Fallback to txt only
        record_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
        with open(record_path, 'w') as f:
            for fp in file_paths:
                f.write(fp + '\n')
        print(f"Saved ingestion record to {record_path} (DB skipped: {e})")

    return combined_df


if __name__ == '__main__':
    merge_multiple_dataframe()
