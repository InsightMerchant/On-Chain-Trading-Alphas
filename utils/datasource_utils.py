import os
import pandas as pd
from utils.interval_helper import extract_interval_from_filename, parse_interval
from utils.timestamp_utils import TIMESTAMP_NAMES  # import the global constant

def aggregate_factor_options(datasource_path):

    datasource_files = [f for f in os.listdir(datasource_path)
                        if f.endswith(".csv") or f.endswith(".parquet")]
    options = []
    ds_map = {}
    native_intervals = []
    
    for ds_file in datasource_files:
        ds_full_path = os.path.join(datasource_path, ds_file)
        try:
            if ds_file.endswith(".parquet"):
                df_tmp = pd.read_parquet(ds_full_path, engine="pyarrow")
            else:
                df_tmp = pd.read_csv(ds_full_path, nrows=1)
        except Exception as e:
            print(f"Error loading {ds_file}: {e}")
            continue

        # Extract the native interval from the file name; default to "1h" if not found.
        ds_interval = extract_interval_from_filename(ds_file)
        if ds_interval is None:
            ds_interval = "1h"
        native_intervals.append(parse_interval(ds_interval))
        
        # Remove columns that appear in TIMESTAMP_NAMES (ignoring case).
        cols = [col for col in df_tmp.columns if col.lower() not in {name.lower() for name in TIMESTAMP_NAMES}]
        
        for col in cols:
            # Format the option as "column_name(file_name)"
            opt = f"{col}({ds_file})"
            options.append(opt)
            ds_map[opt] = (ds_file, col)
    
    return options, ds_map, native_intervals

def merge_datasource_data(data_loader, datasource_path, selected_options, ds_map, interval):

    merged_df = None
    unique_files = set([ds_map[opt][0] for opt in selected_options])
    for ds_file in unique_files:
        ds_df = data_loader.load_datasource_data(ds_file, resample_interval=interval)

        subset_cols = ["datetime"]
        for opt in [o for o in selected_options if ds_map[o][0] == ds_file]:
            # ds_map[opt] returns (ds_file, original_column_name)
            original_col = ds_map[opt][1]
            if original_col in ds_df.columns:
                subset_cols.append(original_col)
        ds_df = ds_df[subset_cols]
        if merged_df is None:
            merged_df = ds_df
        else:
            merged_df = pd.merge(merged_df, ds_df, on="datetime", how="inner")
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


