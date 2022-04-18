import pandas as pd
import numpy as np


def read_data(file_name, training=True):
    if isinstance(file_name, list):
        file_names = file_name
    elif isinstance(file_name, str):
        file_names = [file_name]
    else:
        file_names = []
    df = pd.DataFrame()
    for file in file_names:
        df = pd.concat([df, pd.read_csv(file, index_col=False)], axis=0, ignore_index=True)
    if not training:
        df["label"] = -1
    df = df.sort_values(by=["sn", "fault_time", "label"])
    df["fault_time"] = pd.to_datetime(df["fault_time"], format="%Y-%m-%d %H:%M:%S")
    df['fault_time_ts'] = df["fault_time"].values.astype(np.int64) // 10 ** 9
    return df


def read_log_data(log_file_name):
    if isinstance(log_file_name, list):
        log_file_names = log_file_name
    elif isinstance(log_file_name, str):
        log_file_names = [log_file_name]
    else:
        log_file_names = []

    log_df = pd.DataFrame()
    for file in log_file_names:
        log_df = pd.concat([log_df, pd.read_csv(file, index_col=False)], axis=0, ignore_index=True)
    log_df = log_df.sort_values(by=["sn", "time"])
    log_df["time"] = pd.to_datetime(log_df["time"], format="%Y-%m-%d %H:%M:%S")
    log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
    return log_df

