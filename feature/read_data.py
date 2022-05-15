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


def read_crashdump_data(crash_file_name):
    if isinstance(crash_file_name, list):
        crash_file_names = crash_file_name
    elif isinstance(crash_file_name, str):
        crash_file_names = [crash_file_name]
    else:
        crash_file_names = []

    crash_df = pd.DataFrame()
    for file in crash_file_names:
        crash_df = pd.concat([crash_df, pd.read_csv(file, index_col=False)], axis=0, ignore_index=True)
    crash_df["fault_time"] = pd.to_datetime(crash_df["fault_time"], format="%Y-%m-%d %H:%M:%S")
    # sn,fault_time,fault_code
    return crash_df


def read_venus_data(venus_file_name):
    if isinstance(venus_file_name, list):
        venus_file_names = venus_file_name
    elif isinstance(venus_file_name, str):
        venus_file_names = [venus_file_name]
    else:
        venus_file_names = []

    venus_df = pd.DataFrame()
    for file in venus_file_names:
        venus_df = pd.concat([venus_df, pd.read_csv(file, index_col=False)], axis=0, ignore_index=True)
    venus_df["fault_time"] = pd.to_datetime(venus_df["fault_time"], format="%Y-%m-%d %H:%M:%S")
    # sn,fault_time,module_cause,module
    return venus_df
