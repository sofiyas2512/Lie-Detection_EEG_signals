#import pandas as pd
#import glob
#import os
#
## Update paths and extensions
#lies_folder = 'data/Lie_Sessions'
#truths_folder = 'data/Truth_Sessions'
#
## Use *.csv pattern since files are CSVs
#lie_files = glob.glob(os.path.join(lies_folder, '*.csv'))
#print(f"Found lie CSV files: {lie_files}")
#
#truth_files = glob.glob(os.path.join(truths_folder, '*.csv'))
#print(f"Found truth CSV files: {truth_files}")
#
#dfs_lie = []
#for file in lie_files:
#    print(f"Loading lie file: {file}")
#    df = pd.read_csv(file)
#    print(f"Columns in this lie file: {df.columns.tolist()}")
#    df['label'] = 'lie'
#    dfs_lie.append(df)
#
#dfs_truth = []
#for file in truth_files:
#    print(f"Loading truth file: {file}")
#    df = pd.read_csv(file)
#    print(f"Columns in this truth file: {df.columns.tolist()}")
#    df['label'] = 'truth'
#    dfs_truth.append(df)
#
#print(f"Number of lie DataFrames loaded: {len(dfs_lie)}")
#print(f"Number of truth DataFrames loaded: {len(dfs_truth)}")
#
#if len(dfs_lie) == 0 and len(dfs_truth) == 0:
#    print("No data loaded from either Lie_Sessions or Truth_Sessions folders. Exiting.")
#    exit()
#
#all_data = pd.concat(dfs_lie + dfs_truth, ignore_index=True)
#print(f"Combined dataframe shape: {all_data.shape}")
#
#all_data.to_csv('mock_data.csv', index=False)
#print("Combined CSV saved as mock_data.csv")
#