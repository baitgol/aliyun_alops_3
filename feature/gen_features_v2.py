import os
import pandas as pd
import numpy as np
from scipy import stats
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
from config import model_path, logger

tf_idf_feature_size = 100


features = ['last_category_id',
            'mode_category_id',
            'last_time_gap',
            'last_category_occupy',
            'mode_category_occupy',
            'category_nunique',
            'fault_hour',
            'dup_log_cnt',
            'dup_max_time_gap',
            'dup_mean_time_gap',
            'word_cnt_mean',
            'char_cnt_mean',
            'word_cnt_sum',
            'char_cnt_sum',
            'sn_cnt_of_server_model',
            'sm_encoding_0',
            'sm_encoding_1',
            'sm_encoding_2',
            'sm_encoding_3',
            'sm_encoding_weighted_0',
            'sm_encoding_weighted_1',
            'sm_encoding_weighted_2',
            'sm_encoding_weighted_3'
            ] \
          + ["tfidf_{}".format(i) for i in range(tf_idf_feature_size)] \
#features = ["tfid_{}".format(i) for i in range(tf_idf_feature_size)]
cate_features = ['last_category_id', 'mode_category_id']

def gen_features(df, log_df, training=True):
    logger.info("生成基础特征")
    feature_df = make_base_features(df, log_df)
    # tf-idf
    msg_lst = feature_df["msg"].tolist()

    tf_idf_file = os.path.join(model_path, "tfidf.dat")
    sm_file = os.path.join(model_path, "sever_model_feat.csv")

    if training:
        logger.info("生成tf-idf特征")
        tf_idf_df, tf_idf = gen_tf_idf_features(msg_lst, None)
        pickle.dump(tf_idf, open(tf_idf_file, "wb"))
        logger.info("保存tf-idf模型成功")
        sm_df = make_server_model_target_encoding(feature_df)
        sm_df.to_csv(sm_file, index=False)
        logger.info("保存sever_model_feat模型成功")
    else:

        loaded_tf_idf = pickle.load(open(tf_idf_file, "rb"))
        logger.info("加载tf-idf模型成功")
        tf_idf_df, _ = gen_tf_idf_features(msg_lst, loaded_tf_idf)
        logger.info("生成tf-idf特征")
        sm_df = pd.read_csv(sm_file, index_col=False)
        logger.info("加载sever_model_feat模型成功")

    feature_df = pd.concat([feature_df, tf_idf_df], axis=1)
    feature_df = feature_df.merge(sm_df, how="left")

    # feature_df = make_final_features(feature_df, crash_df, venus_df, training)

    logger.info("生成特征成功")
    return feature_df


def make_base_features(df, log_df):

    # df
    df["id"] = df["sn"] + df["fault_time_ts"].map(str) + df["label"].map(str)

    # 关联日志, 基础字段处理
    all_df = df.merge(log_df[["sn", "time", "time_ts", "msg", "server_model"]], on=["sn"])
    all_df = all_df.loc[all_df["time"] <= all_df["fault_time"]]
    all_df["msg"] = all_df["msg"].map(lambda x: x.replace("|", ""))

    all_df["category"] = all_df["msg"].map(lambda x: x.strip().split(" ")[0])
    all_df["category_id"] = all_df["category"].map(lambda x: cate_map.get(x, len(cate_map)))
    all_df['word_cnt'] = all_df['msg'].apply(lambda x: len(str(x).split(" ")))
    all_df['char_cnt'] = all_df['msg'].str.len()


    # 去重
    all_df = all_df.sort_values(by=["id", "time"])
    all_df = all_df.drop_duplicates(subset=["id", "msg"], keep="last")

    msg_df = all_df.groupby(["id", "label"]).agg({"msg": lambda x: "。".join(x)}).reset_index()
    # msg_df.loc[msg_df.label<=1].to_csv("train_msg.csv", index=False)

    all_df = all_df.groupby("id").head(40)
    # tail_df = all_df.groupby("id").tail(20)

    #all_df = pd.concat([head_df, tail_df], ignore_index=True)
    #all_df = all_df.sort_values(by=["id", "time"])
    #all_df = all_df.drop_duplicates(keep="last")


    cate_feat_df = all_df.groupby("id").agg({"category_id": ["last", lambda x: stats.mode(x)[0][0]]}).reset_index()
    cate_feat_df.columns = ["id", "last_category_id", "mode_category_id"]
    all_df = all_df.merge(cate_feat_df, on=["id"])

    all_df["is_last_category"] = np.where(all_df["category_id"] == all_df["last_category_id"], 1, 0)
    all_df["is_mode_category"] = np.where(all_df["category_id"] == all_df["mode_category_id"], 1, 0)

    # 时间
    time_feat_df = all_df.groupby(["id", "fault_time", "fault_time_ts"]).agg({"time_ts": ["size", "min", "max"]}).reset_index()
    time_feat_df.columns = ["id", "fault_time", "fault_time_ts", "dup_log_cnt", "min_time_ts", "max_time_ts"]

    time_feat_df["dup_max_time_gap"] = time_feat_df["fault_time_ts"] - time_feat_df["min_time_ts"]
    time_feat_df["dup_mean_time_gap"] = time_feat_df["dup_max_time_gap"] / time_feat_df["dup_log_cnt"]
    time_feat_df["last_time_gap"] = time_feat_df["fault_time_ts"] - time_feat_df["max_time_ts"]
    time_feat_df["fault_hour"] = time_feat_df["fault_time"].dt.hour

    # tail

    #msg_df = all_df.groupby(["id"]).agg({"msg":lambda x: " ".join(x)})
    feat_df = all_df.groupby(["id", "sn", "fault_time","fault_time_ts", "label", "server_model",
                                 "last_category_id", "mode_category_id"]).agg({"category_id": "nunique", "is_last_category":"mean",
                                        "is_mode_category":"mean", "word_cnt": ["mean", "sum"],
                                        "char_cnt": ["mean", "sum"], "time_ts": ["size", "min"]}).reset_index()
    feat_df.columns = ["id",  "sn", "fault_time", "fault_time_ts", "label", "server_model", "last_category_id",
                       "mode_category_id", "category_nunique", "last_category_occupy",
                       "mode_category_occupy", "word_cnt_mean", "word_cnt_sum", "char_cnt_mean", "char_cnt_sum",
                       "tail_log_cnt", "min_time_ts"]

    feat_df["tail_max_time_gap"] = feat_df["fault_time_ts"] - feat_df["min_time_ts"]
    feat_df["tail_mean_time_gap"] = feat_df["tail_max_time_gap"] / feat_df["tail_log_cnt"]



    feat_df = feat_df.merge(time_feat_df[["id", "dup_log_cnt", "dup_max_time_gap", "dup_mean_time_gap", "last_time_gap", "fault_hour"]],
                            on=["id"])


    feat_df = feat_df.merge(msg_df, how="left",  on=['id', 'label'])

    # print(feat_df.columns)
    # print(feat_df)
    return feat_df


def make_server_model_target_encoding(df):
    target_mean_dict = df["label"].value_counts(normalize=True).to_dict()

    se_df = df.groupby(["server_model"]).agg({"sn": ["size", "nunique"]}).reset_index()
    se_df.columns = ["server_model", "cnt", "sn_cnt_of_server_model"]

    se_df_1 = df.groupby(["server_model", "label"]).agg({"sn": "size"}).reset_index()
    se_df_1 = pd.pivot_table(se_df_1, values='sn', columns='label', index='server_model', fill_value=0)
    se_df_1.columns = ['sm_encoding_0', 'sm_encoding_1', 'sm_encoding_2', 'sm_encoding_3']
    se_df_1 = se_df_1.reset_index()
    se_df = se_df.merge(se_df_1)
    for i in range(4):
        se_df[f"sm_encoding_{i}"] = se_df[f"sm_encoding_{i}"] / se_df["cnt"]
        se_df[f"sm_encoding_weighted_{i}"] = se_df[f"sm_encoding_{i}"] / se_df["cnt"] / target_mean_dict[i]

    se_df["sm_encoding_weighted_sum"] = se_df["sm_encoding_weighted_0"] + \
                                        se_df["sm_encoding_weighted_1"] + se_df["sm_encoding_weighted_2"] + se_df[
                                            "sm_encoding_weighted_3"]
    for i in range(4):
        se_df[f"sm_encoding_weighted_{i}"] = se_df[f"sm_encoding_weighted_{i}"] / se_df["sm_encoding_weighted_sum"]

    return se_df

def gen_tf_idf_features(msg_lst, tf_idf=None):

    if tf_idf is None:
        tf_idf = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), stop_words='english',
                            max_features=tf_idf_feature_size)
        tf_idf_df = tf_idf.fit_transform(msg_lst).todense()
        tf_idf_df = pd.DataFrame(tf_idf_df,
                                      columns=["tfidf_{}".format(i) for i in range(tf_idf_feature_size)])
    else:
        tf_idf_df = tf_idf.transform(msg_lst).todense()
        tf_idf_df = pd.DataFrame(tf_idf_df, columns=["tfidf_{}".format(i) for i in range(tf_idf_feature_size)])
    return tf_idf_df, tf_idf


cate_map = {
    'Memory': 0,
    'System': 1,
    'Processor': 2,
    'Temperature': 3,
    'Drive': 4,
    'Power': 5,
    'Unknown': 6,
    'Microcontroller': 7,
    'OS': 8,
    'Watchdog2': 9,
    'OEM': 10,
    'Button': 11,
    'Slot/Connector': 12,
    'Microcontroller/Coprocessor': 13,
    'Management': 14,
    'Event': 15,
    'Watchdog': 16,
    'Slot': 17,
    'Fan': 18,
    'Critical': 19,
    'device': 20,
    'LAN': 21,
    'Version': 22,
    'Add-in': 23,
    'Terminator': 24,
    'Chassis': 25,
    'reserved': 26,
    'Physical': 27,
    'Session': 28,
    'Reserved': 29,
    'Cable/Interconnect': 30,
    'Cable': 31,
    'Chip': 32,
    'Battery': 33
}



