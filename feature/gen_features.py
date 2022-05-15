import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec
import pickle
import logging
from scipy import stats
from config import model_path, logger
from sklearn.model_selection import StratifiedKFold

tf_idf_feature_size = 100

features = ['last_category_id',
            'mode_category_id',
            'log_cnt',
            'last_category_occupy',
            'mode_category_occupy',
            'category_nunique',
            'last_time_gap', 'max_time_gap',
            'sn_cnt_of_server_model',
            'sm_encoding_0',
            'sm_encoding_1',
            'sm_encoding_2',
            'sm_encoding_3',
            'sm_encoding_weighted_0',
            'sm_encoding_weighted_1',
            'sm_encoding_weighted_2',
            'sm_encoding_weighted_3',
            'mean_time_gap',
            'word_cnt_sum',
            'char_cnt_sum',
            'fault_hour'

            ] + [f"tfidf_{i}" for i in range(tf_idf_feature_size)]


def gen_features(df, log_df, training=True):
    logger.info("生成基础特征")
    feature_df = make_base_features(df, log_df)
    # tf-idf
    msg_lst = feature_df["msg"].tolist()

    tf_idf_file = os.path.join(model_path, "tfidf.dat")
    w2v_file = os.path.join(model_path, 'w2v.dat')
    sm_file = os.path.join(model_path, "sever_model_feat.csv")

    if training:
        logger.info("生成tf-idf特征")
        tf_idf_df, tf_idf = gen_tf_idf_features(msg_lst, None)
        pickle.dump(tf_idf, open(tf_idf_file, "wb"))
        logger.info("保存tf-idf模型成功")

        # w2v_df, w2v_model = gen_word2vec_features(msg_lst, None)
        # pickle.dump(w2v_model, open(w2v_file, "wb"))
        # logger.info("保存word2vec模型成功")

        sm_df = make_server_model_target_encoding(feature_df)
        sm_df.to_csv(sm_file, index=False)
        logger.info("保存sever_model_feat模型成功")
    else:

        loaded_tf_idf = pickle.load(open(tf_idf_file, "rb"))
        logger.info("加载tf-idf模型成功")
        tf_idf_df, _ = gen_tf_idf_features(msg_lst, loaded_tf_idf)
        logger.info("生成tf-idf特征")

        # loaded_w2v = pickle.load(open(w2v_file, "rb"))
        # logger.info("加载word2vec模型成功")
        # w2v_df, _ = gen_word2vec_features(msg_lst, loaded_w2v)
        # logger.info("生成word2vec特征")

        sm_df = pd.read_csv(sm_file, index_col=False)
        logger.info("加载sever_model_feat模型成功")
    feature_df = pd.concat([feature_df, tf_idf_df], axis=1)
    feature_df = feature_df.merge(sm_df, how="left")

    #feature_df = make_final_features(feature_df, crash_df, venus_df, training)

    logger.info("生成特征成功")
    return feature_df

def make_base_features(df, log_df):
    # 处理日期
    df["fault_time"] = pd.to_datetime(df["fault_time"], format="%Y-%m-%d %H:%M:%S")
    df['fault_time_ts'] = df["fault_time"].values.astype(np.int64) // 10 ** 9

    df = df.merge(log_df[['sn', 'time', 'msg', 'server_model']], how="left", on=["sn"])
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df['time_ts'] = df["time"].values.astype(np.int64) // 10 ** 9
    df = df.loc[(df.fault_time >= df.time)]

    df = df.sort_values(by=["sn", "fault_time", "label", "time"])

    df = df.drop_duplicates(subset=["sn", "fault_time", "label", "msg"], keep="last")
    df["msg"] = df["msg"].map(lambda x: x.replace("|", ""))

    sub_df = df.groupby(["sn", "fault_time", "label"]).head(40)
    # sub_df = pd.concat([sub_df1, sub_df2], axis=0, ignore_index=True)

    sub_df["category"] = sub_df["msg"].map(lambda x: x.strip().split(" ")[0])
    sub_df["category_id"] = sub_df["category"].map(lambda x: cate_map.get(x, len(cate_map)))

    sub_df['word_cnt'] = sub_df['msg'].apply(lambda x: len(str(x).split(" ")))
    sub_df['char_cnt'] = sub_df['msg'].str.len()

    tmp_df = sub_df.drop_duplicates(subset=["sn", "fault_time", "label"], keep="last")
    tmp_df = tmp_df[["sn", "fault_time", "label", "category_id"]]
    tmp_df.columns = ["sn", "fault_time", "label", "last_category_id"]
    sub_df = sub_df.merge(tmp_df, on=["sn", "fault_time", "label"])

    sub_df["is_last_category"] = np.where(sub_df["category_id"] == sub_df["last_category_id"], 1, 0)

    tmp_df = sub_df.groupby(["sn", "fault_time", "label"]).agg(
        {"category_id": lambda x: stats.mode(x)[0][0]}).reset_index()
    tmp_df.columns = ["sn", "fault_time", "label", "mode_category_id"]

    sub_df = sub_df.merge(tmp_df, on=["sn", "fault_time", "label"])
    sub_df["is_mode_category"] = np.where(sub_df["category_id"] == sub_df["mode_category_id"], 1, 0)

    feat_df = sub_df.groupby(["sn", "fault_time", "fault_time_ts", "label",
                              "last_category_id", "mode_category_id"]).agg({"msg": "size",
                                                                            "is_last_category": "sum",
                                                                            "is_mode_category": "sum",
                                                                            "category_id": "nunique",
                                                                            "word_cnt": "mean",
                                                                            "char_cnt": "mean",
                                                                            "time_ts": ["min", "max"]}).reset_index()

    feat_df.columns = ["sn", "fault_time", "fault_time_ts", "label", "last_category_id", "mode_category_id",
                       "log_cnt", "last_category_sum", "mode_category_sum", "category_nunique", "word_cnt", "char_cnt",
                       "time_ts_min", "time_ts_max"]
    feat_df["last_category_occupy"] = feat_df["last_category_sum"] / feat_df["log_cnt"]
    feat_df["mode_category_occupy"] = feat_df["mode_category_sum"] / feat_df["log_cnt"]

    feat_df["last_time_gap"] = feat_df["fault_time_ts"] - feat_df["time_ts_max"]
    feat_df["max_time_gap"] = feat_df["fault_time_ts"] - feat_df["time_ts_min"]

    feat_df = feat_df[['sn', 'fault_time', 'label', "last_category_id", "mode_category_id",
                       'log_cnt', 'last_category_occupy', 'mode_category_occupy', 'category_nunique',
                       'last_time_gap', 'max_time_gap', 'word_cnt', 'char_cnt']]

    feat_df["mean_time_gap"] = feat_df["max_time_gap"] / feat_df["log_cnt"]

    feat_df["word_cnt_sum"] = feat_df["word_cnt"] * feat_df["log_cnt"]
    feat_df["char_cnt_sum"] = feat_df["char_cnt"] * feat_df["log_cnt"]

    # 合并日志
    df = df.groupby(["sn", "fault_time", "label", "server_model"])["msg"].agg({lambda x: " ".join(x)}).reset_index()
    df.columns = ["sn", "fault_time", "label", "server_model", "msg"]

    df = df.merge(feat_df, on=['sn', 'fault_time', 'label'], how="left")

    # 故障时间
    df["fault_hour"] = df["fault_time"].dt.hour

    # print(len(result_lst))
    return df


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
        tf_idf.fit(msg_lst)

    tf_idf_df = tf_idf.transform(msg_lst).todense()
    tf_idf_df = pd.DataFrame(tf_idf_df,
                                 columns=[f"tfidf_{i}" for i in range(tf_idf_feature_size)])
    return tf_idf_df, tf_idf

#
def gen_word2vec_features(msg_lst, w2v_model=None):
    if w2v_model is None:
        w2v_model = Word2Vec(msg_lst, vector_size=32, window=5, min_count=5, sg=1, hs=0, seed=2022)

    sentence_emb = []
    for sent in msg_lst:
        vec = []
        for w in sent:
            if w in w2v_model.wv:
                vec.append(w2v_model.wv[w])
        if len(vec) > 0:
            sentence_emb.append(np.mean(vec, axis=0))
        else:
            sentence_emb.append([0] * 32)
    w2v_df = pd.DataFrame(sentence_emb, columns=[f"w2v_{i}" for i in range(32)])
    return w2v_df, w2v_model

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



