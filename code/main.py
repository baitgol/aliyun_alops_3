import numpy as np
import pandas as pd
import random
import pickle

from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

from read_data import read_data, read_log_data
from gen_features import gen_features, features, cate_features
from metrics import macro_f1


np.random.seed(0)
random.seed(0)

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def main():
    #train()
    predict()

def get_class_weights(train_label):
    classes = np.unique(train_label)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_label)
    class_weights = dict(zip(classes, weights))
    return class_weights

def train():

    logging.info("开始训练.......")
    train_file_names = ["../data/preliminary_train_label_dataset.csv",
                        "../data/preliminary_train_label_dataset_s.csv"]
    train_df = read_data(train_file_names)
    # 去重
    train_df = train_df.drop_duplicates(subset=["sn", "fault_time", "label"])
    train_log_file = "../data/preliminary_sel_log_dataset.csv"
    train_log_df = read_log_data(train_log_file)
    logging.info("读取数据成功.........")
    logging.info("开始生成特征.........")

    #train_feature_df = gen_features(train_df.iloc[:], train_log_df)
    # 保存
   # train_feature_df.to_csv("user_data/tmp_data/train_features_final.csv", index=False)
    train_feature_df = pd.read_csv("user_data/tmp_data/train_features_final.csv", index_col=False)

    train_features = train_feature_df[features]
    train_label = train_feature_df["label"]
    class_weights = get_class_weights(train_label)

    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=2010)
    f1_scores = []
    true_train_lst = []
    pred_train_lst = []
    true_valid_lst = []
    pred_valid_lst = []
    logging.info("开始生成模型,5折交叉验证.........")
    for idx, (train_idx, valid_idx) in enumerate(kflod.split(train_features.iloc[:], train_label)):
        X_train = train_features.iloc[train_idx]
        y_train = train_label[train_idx]
        #     sample_solver = SMOTE(random_state=0)
        #     X_sample ,y_sample = sample_solver.fit_resample(X_train,y_train)
        X_valid = train_features.iloc[valid_idx]
        y_valid = train_label[valid_idx]

        gbm = LGBMClassifier(objective="multiclass",
                             n_estimators=100,
                             learning_rate=0.1,
                             importance_type="gain",
                             class_weight=class_weights)
        gbm.fit(X_train, y_train, categorical_feature=cate_features)

        pickle.dump(gbm, open(f"user_data/model_data/gbm_model_{idx}.dat", "wb"))
        pred_valid = gbm.predict(X_valid)

        f1_score = macro_f1(y_valid.values, pred_valid)
        logging.info(f"{idx + 1}/5 f1-score:{np.round(f1_score, 4)}" )
        f1_scores.append(f1_score)

        true_valid_lst.extend(y_valid.to_list())
        pred_valid_lst.extend(list(pred_valid))

        pred_train = gbm.predict(X_train)
        true_train_lst.extend(y_train.to_list())
        pred_train_lst.extend(list(pred_train))

    logging.info(f"train macro f1-score:{np.round(macro_f1(true_train_lst, pred_train_lst), 4)}")
    logging.info(f"valid macro f1-socore: {np.round(macro_f1(true_valid_lst, pred_valid_lst), 4)}")

    logging.info("训练完成....................")


def predict():
    logging.info("开始预测....................")
    logging.info("读取测试数据")
    # test_file_name = "data/final_submit_dataset_a.csv"
    # test_log_file = "data/final_sel_log_dataset_a.csv"

    test_file_name = "tcdata/final_submit_dataset_a.csv"
    test_log_file = "tcdata/final_sel_log_dataset_a.csv"

    logging.info(f"test_file_name={test_file_name}")
    test_df = read_data(test_file_name, False)
    logging.info(f"测试数据数据量={len(test_df)}")
    test_df["label"] = -1
    logging.info(f"test_log_file={test_log_file}")
    logging.info("读取测试日志")
    test_log_df = read_log_data(test_log_file)

    test_feature_df = gen_features(test_df, test_log_df, training=False)
    # 保存
    test_feature_df.to_csv("user_data/tmp_data/test_features_final.csv", index=False)

    #test_feature_df = pd.read_csv("../user_data/tmp_data/test_features_final.csv", index_col=False)
    test_features = test_feature_df[features]
    test_pred = np.zeros((len(test_feature_df), 4))
    logging.info("开始读取模型")
    for idx in range(5):
        model_name = f"user_data/model_data/gbm_model_{idx}.dat"
        logging.info(f"load model, model_name={model_name}")
        loaded_model = pickle.load(open(model_name, "rb"))
        logging.info(f"加载模型{idx+1}")
        test_pred_prob = loaded_model.predict_proba(test_features[:])
        test_pred += test_pred_prob

    test_feature_df["label"] = test_pred.argmax(1)
    test_feature_df[["sn", "fault_time", "label"]].to_csv("prediction_result/final_pred_a.csv", index=False)
    logging.info("预测完成")

if __name__ == "__main__":
    main()