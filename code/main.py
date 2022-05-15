import os
import numpy as np
import pandas as pd
import random
import pickle

import warnings
warnings.filterwarnings("ignore")
from config import project_path, model_path, logger
from feature.read_data import read_data, read_log_data
from feature.gen_features_v2 import gen_features, features
#from model.cv_model import cv_model_train, cv_model_predict, cv_model_eval
from model.cv_model_pseudo import cv_model_train, cv_model_predict, cv_model_eval
from model.utils import macro_f1
np.random.seed(0)
random.seed(0)



def main():
    # train()
    # eval()
    predict()

#
def eval():
    logger.info("开始验证......")

    train_feature_df = pd.read_csv(os.path.join(project_path, "user_data/tmp_data/train_features_final.csv"),
                                   index_col=False)

    X_train = train_feature_df[features]
    y_train = train_feature_df["label"]

    lgb_oof_pred = cv_model_eval(X_train, y_train, model_name="lgb")
    xgb_oof_pred = cv_model_eval(X_train, y_train, model_name="xgb")

    oof_preds = 0.5 * lgb_oof_pred + 0.5 * xgb_oof_pred
    logger.info(f"******lgb and xgb ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}*******")
    # cat_oof_pred = cv_model_eval(X_train, y_train, model_name="cat")
    # oof_preds = (lgb_oof_pred + xgb_oof_pred + cat_oof_pred) / 3
    # logger.info(f"*****lgb and xgb and cat ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}*****")


def get_pseudo_data():
    logger.info("读取unlabeled数据")
    test_file_name = [os.path.join(project_path, "data/preliminary_submit_dataset_a.csv"),
                      os.path.join(project_path, "data/preliminary_submit_dataset_b.csv")]
    test_log_file = [os.path.join(project_path, "data/preliminary_sel_log_dataset_a.csv"),
                     os.path.join(project_path, "data/preliminary_sel_log_dataset_b.csv")]

    test_df = read_data(test_file_name, False)
    logger.info(f"unlabeled数据数据量={len(test_df)}")
    test_df["label"] = -1
    logger.info("读取unlabeled日志")
    test_log_df = read_log_data(test_log_file)
    unlabeled_feature_df = gen_features(test_df, test_log_df, training=False)

    return unlabeled_feature_df



def train():

    logger.info("开始训练.......")
    train_file_names = [os.path.join(project_path, "data/preliminary_train_label_dataset.csv"),
                        os.path.join(project_path, "data/preliminary_train_label_dataset_s.csv")]
    train_df = read_data(train_file_names)
    # 去重
    train_df = train_df.drop_duplicates(subset=["sn", "fault_time", "label"])
    # log
    train_log_file = os.path.join(project_path, "data/preliminary_sel_log_dataset.csv")
    train_log_df = read_log_data(train_log_file)

    logger.info("读取数据成功.........")
    logger.info("开始生成特征.........")

    train_feature_df = gen_features(train_df.iloc[:], train_log_df, True)
    # 保存

    train_feature_df.to_csv(os.path.join(project_path, "user_data/tmp_data/train_features_final.csv"), index=False)
    #train_feature_df = pd.read_csv(os.path.join(project_path, "user_data/tmp_data/train_features_final.csv"), index_col=False)

    X_train = train_feature_df[features]
    y_train = train_feature_df["label"]


    # lgb_oof_pred = cv_model_train(X_train, y_train, model_name="lgb")
    # xgb_oof_pred = cv_model_train(X_train, y_train, model_name="xgb")
    # oof_preds = 0.5 * lgb_oof_pred + 0.5 * xgb_oof_pred
    # logger.info(f"lgb and xgb ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}")
    # cat_oof_pred = cv_model_train(X_train, y_train, model_name="cat")
    # oof_preds = (lgb_oof_pred + xgb_oof_pred + cat_oof_pred) / 3
    # logger.info(f"lgb and xgb and cat ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}")
    # oof_preds = 0.4 * lgb_oof_pred + 0.4 * xgb_oof_pred + 0.2 * cat_oof_pred
    # logger.info(f"lgb and xgb and cat ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}")

    #pesudo-伪标签
    unlabeled_feature_df = get_pseudo_data()
    unlabeled_train = unlabeled_feature_df[features]
    lgb_oof_pred = cv_model_train(X_train, y_train, unlabeled_train, model_name="lgb")
    xgb_oof_pred = cv_model_train(X_train, y_train, unlabeled_train, model_name="xgb")
    oof_preds = 0.5 * lgb_oof_pred + 0.5 * xgb_oof_pred
    logger.info(f"lgb and xgb ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}")
    cat_oof_pred = cv_model_train(X_train, y_train, unlabeled_train, model_name="cat")
    oof_preds = (lgb_oof_pred + xgb_oof_pred + cat_oof_pred) / 3
    logger.info(f"lgb and xgb and cat ensemble scores: {macro_f1(y_train, oof_preds.argmax(1))}")


def predict():
    logger.info("开始预测....................")
    logger.info("读取测试数据")
    # test_file_name = os.path.join(project_path, "data/preliminary_submit_dataset_b.csv")
    # test_log_file = os.path.join(project_path, "data/preliminary_sel_log_dataset_b.csv")


    test_file_name = "tcdata/final_submit_dataset_b.csv"
    test_log_file = "tcdata/final_sel_log_dataset_b.csv"

    logger.info(f"test_file_name={test_file_name}")
    test_df = read_data(test_file_name, False)
    logger.info(f"测试数据数据量={len(test_df)}")
    test_df["label"] = -1
    logger.info(f"test_log_file={test_log_file}")
    logger.info("读取测试日志")
    test_log_df = read_log_data(test_log_file)
    test_feature_df = gen_features(test_df, test_log_df, training=False)

    # 保存
    # test_feature_df.to_csv(os.path.join(project_path, "user_data/tmp_data/test_features_final.csv"), index=False)
    # test_feature_df = pd.read_csv("../user_data/tmp_data/test_features_final.csv", index_col=False)

    lgb_test_pred = cv_model_predict(test_feature_df[features], model_name="lgb")
    xgb_test_pred = cv_model_predict(test_feature_df[features], model_name="xgb")
    cat_test_pred = cv_model_predict(test_feature_df[features], model_name="cat")
    test_pred = lgb_test_pred + xgb_test_pred + cat_test_pred
    test_feature_df["label"] = test_pred.argmax(1)
    test_feature_df[["sn", "fault_time", "label"]].to_csv(
        os.path.join(project_path, "prediction_result/final_pred_b.csv"), index=False)
    logger.info("预测完成")


if __name__ == "__main__":
    main()