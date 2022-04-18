import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
tf_idf_feature_size = 100


features = ['last_category_id',
            'last_time_gap',
            'fault_hour',
            'sn_cnt_of_server_model',
            'sm_encoding_0',
            'sm_encoding_1',
            'sm_encoding_2',
            'sm_encoding_3',
            'sm_encoding_weighted_0',
            'sm_encoding_weighted_1',
            'sm_encoding_weighted_2',
            'sm_encoding_weighted_3'] + [prefix+"_"+feat for prefix in ["total", "dup", "tail"] for feat in ['log_cnt','mode_category_id', 'last_category_occupy','mode_category_occupy',
            'category_nunique','max_time_gap', 'mean_time_gap', 'word_cnt_mean', 'char_cnt_mean', 'word_cnt_sum','char_cnt_sum']] \
           + ["tfid_{}".format(i) for i in range(tf_idf_feature_size)]
cate_features = ['last_category_id', 'total_mode_category_id', 'dup_mode_category_id','tail_mode_category_id']

def gen_features(df, log_df, training=True):
    logging.info("生成基础特征")
    feature_df = make_base_features(df, log_df)
    # tf-idf
    msg_lst = feature_df["msg"].tolist()
    if training:
        logging.info("生成tf-idf特征")
        tf_idf_df, tf_idf = gen_tf_idf_features(msg_lst, None)
        pickle.dump(tf_idf, open("user_data/model_data/tfidf.dat", "wb"))
        logging.info("保存tf-idf模型成功")
        sm_df = make_server_model_target_encoding(feature_df)
        sm_df.to_csv("user_data/model_data/sever_model_feat.csv", index=False)
        logging.info("保存sever_model_feat模型成功")
    else:
        tf_idf_model_file = "user_data/model_data/tfidf.dat"
        loaded_tf_idf = pickle.load(open(tf_idf_model_file, "rb"))
        logging.info("加载tf-idf模型成功")
        tf_idf_df, _ = gen_tf_idf_features(msg_lst, loaded_tf_idf)
        logging.info("生成tf-idf特征")
        sm_df = pd.read_csv("user_data/model_data/sever_model_feat.csv", index_col=False)
        logging.info("加载sever_model_feat模型成功")
    feature_df = pd.concat([feature_df, tf_idf_df], axis=1)
    feature_df = feature_df.merge(sm_df, how="left")
    logging.info("生成特征成功")
    return feature_df


def get_sub_base_features(fault_time_ts, sel_data, prefix="total"):
    log_cnt = len(sel_data)
    max_time_gap = fault_time_ts - sel_data["time_ts"].min()
    mean_time_gap = max_time_gap / log_cnt

    sel_data['word_cnt'] = sel_data['msg'].apply(lambda x: len(str(x).split(" ")))
    sel_data['char_cnt'] = sel_data['msg'].str.len()
    word_cnt_mean = sel_data['word_cnt'].mean()
    char_cnt_mean = sel_data["char_cnt"].mean()
    word_cnt_sum = sel_data['word_cnt'].sum()
    char_cnt_sum = sel_data['char_cnt'].sum()

    category_nunique = sel_data["category"].nunique()
    # last category
    last_category_occupy = sel_data["is_last_category"].sum() / len(sel_data)
    # mode category
    mode_category_id = sel_data["category_id"].value_counts().index[0]
    sel_data["is_mode_category"] = np.where(sel_data["category_id"] == mode_category_id, 1, 0)
    mode_category_occupy = sel_data["is_mode_category"].sum() / len(sel_data)

    # print(msg)
    ret_dict = {
                 prefix+"_"+"log_cnt": log_cnt,
                 prefix+"_"+"mode_category_id": mode_category_id,
                 prefix+"_"+"last_category_occupy": last_category_occupy,
                 prefix+"_"+"mode_category_occupy": mode_category_occupy,
                 prefix+"_"+"category_nunique": category_nunique,
                 prefix+"_"+"max_time_gap": max_time_gap,
                 prefix+"_"+"mean_time_gap": mean_time_gap,
                 prefix+"_"+"word_cnt_mean": word_cnt_mean,
                 prefix+"_"+"char_cnt_mean": char_cnt_mean,
                 prefix+"_"+"word_cnt_sum": word_cnt_sum,
                 prefix+"_"+"char_cnt_sum": char_cnt_sum,

                }

    return ret_dict
def make_base_features(df, log_df, traning=True):
    # 处理日期
    result_lst = []
    for idx, row in tqdm(df.iterrows()):
        sn = row["sn"]
        fault_time = row["fault_time"]
        fault_time_ts = row["fault_time_ts"]

        label = row["label"]
        fault_hour = fault_time.hour

        sel_data = log_df.loc[(log_df["sn"] == sn) & (log_df["time"] <= fault_time)]
        ret_dict = {"sn": sn,
                    "fault_time": fault_time,
                    "label": label,
                    "fault_hour": fault_hour
                    }
        if not sel_data.empty:

            sel_data["fault_time"] = fault_time
            sel_data = sel_data.sort_values(by=["time"])
            sel_data["msg"] = sel_data["msg"].map(lambda x: x.replace("|", ""))

            last_time_gap = fault_time_ts - sel_data["time_ts"].max()
            server_model = sel_data.iloc[0]["server_model"]

            sel_data["category"] = sel_data["msg"].map(lambda x: x.strip().split(" ")[0])
            sel_data["category_id"] = sel_data["category"].map(lambda x: cate_map.get(x, len(cate_map)))
            last_category_id = sel_data.tail(1)["category_id"].values[0]
            sel_data["is_last_category"] = np.where(sel_data["category_id"] == last_category_id, 1, 0)

            ret_dict.update({
                        "server_model": server_model,
                        "last_time_gap": last_time_gap,
                        "last_category_id": last_category_id
                        })
            # 不去重
            ret_dict.update(get_sub_base_features(fault_time_ts, sel_data, prefix="total"))
            # 去重
            sel_data = sel_data.drop_duplicates(subset=["msg"], keep="last")
            ret_dict.update(get_sub_base_features(fault_time_ts, sel_data, prefix="dup"))
            # msg_list
            msg = " ".join(sel_data["msg"].tolist())
            ret_dict["msg"] = msg
            # 取后面40条日志
            sel_data_tail = sel_data.tail(40)
            ret_dict.update(get_sub_base_features(fault_time_ts, sel_data_tail, prefix="tail"))

            # # 取前面40条日志
            # sel_data_head = sel_data.head(40)
            # ret_dict.update(get_sub_base_features(fault_time_ts, sel_data_head, prefix="head"))
            #
            # # 取前后各20条日志
            # sel_data_head = sel_data.head(20)
            # sel_data_tail = sel_data.tail(20)
            #
            # sel_data_ht = pd.concat([sel_data_head, sel_data_tail], ignore_index=True)
            # sel_data_ht = sel_data_ht.drop_duplicates(subset=["msg"])
            # ret_dict.update(get_sub_base_features(fault_time_ts, sel_data_ht, prefix="ht"))
            # print(ret_dict.keys())
            result_lst.append(ret_dict)
        elif traning:
            continue
        else:
            ret_dict['msg'] = " "
            result_lst.append(ret_dict)


    # print(len(result_lst))
    return pd.DataFrame(result_lst)


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
                                      columns=["tfid_{}".format(i) for i in range(tf_idf_feature_size)])
    else:
        tf_idf_df = tf_idf.transform(msg_lst).todense()
        tf_idf_df = pd.DataFrame(tf_idf_df, columns=["tfid_{}".format(i) for i in range(tf_idf_feature_size)])
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



