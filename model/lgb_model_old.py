import os
import numpy as np
import pandas as pd
import random
import pickle

from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from config import project_path, model_path, logger
from feature.gen_features_v2 import cate_features
from model.utils import macro_f1

np.random.seed(0)
random.seed(0)


def get_class_weights(train_label):
    classes = np.unique(train_label)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_label)
    class_weights = dict(zip(classes, weights))
    return class_weights

def lgb_train(X_train, y_train):
    class_weights = get_class_weights(y_train)
    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=2010)
    f1_scores = []

    true_tr_lst = []
    pred_tr_lst = []
    true_val_lst = []
    pred_val_lst = []
    logger.info("开始生成模型,5折交叉验证.........")
    for idx, (train_idx, valid_idx) in enumerate(kflod.split(X_train.iloc[:], y_train)):
        x_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]

        X_val = X_train.iloc[valid_idx]
        y_val = y_train[valid_idx]

        gbm = LGBMClassifier(objective="multiclass",
                             n_estimators=100,
                             learning_rate=0.1,
                             importance_type="gain",
                             class_weight=class_weights)
        gbm.fit(x_tr, y_tr, categorical_feature=cate_features)

        pred_tr = gbm.predict_proba(x_tr).argmax(1)
        true_tr_lst.extend(y_tr.to_list())
        pred_tr_lst.extend(list(pred_tr))


        model_file = os.path.join(model_path, f"gbm_model_{idx}.dat")
        pickle.dump(gbm, open(model_file, "wb"))


        pred_valid = gbm.predict_proba(X_val).argmax(1)
        val_f1_score = macro_f1(y_val.values, pred_valid)
        logger.info(f"{idx + 1}/5 f1-score:{np.round(val_f1_score, 4)}")
        f1_scores.append(val_f1_score)
        true_val_lst.extend(y_val.to_list())
        pred_val_lst.extend(list(pred_valid))

    logger.info(f"train macro f1-score:{np.round(macro_f1(true_tr_lst, pred_tr_lst), 4)}")
    logger.info(f"valid macro f1-socore: {np.round(macro_f1(true_val_lst, pred_val_lst), 4)}")
    logger.info("训练完成....................")


def lgb_predict(X_test):
    test_pred = np.zeros((len(X_test), 4))
    logger.info("开始读取模型")
    for idx in range(5):
        model_file = os.path.join(model_path, f"gbm_model_{idx}.dat")

        logger.info(f"load model, model_name={model_file}")
        loaded_model = pickle.load(open(model_file, "rb"))
        logger.info(f"加载模型{idx + 1}")
        test_pred_prob = loaded_model.predict_proba(X_test)
        test_pred += test_pred_prob

    return test_pred.argmax(1)



