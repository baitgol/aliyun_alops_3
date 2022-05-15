import numpy as np
import os
import pickle
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostClassifier
from scipy.misc import derivative
from feature.gen_features_v2 import cate_features
from config import project_path, model_path, logger
from model.lgb_model import lgb_model_train
from model.xgb_model import xgb_model_train
from model.cat_model import cat_model_train
from model.utils import macro_f1, get_class_weights, dump_model, load_model




def cv_model_stacking_train(X_train, y_train):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2010)
    num_class = 4
    num_model = 3

    train_oof_predicts = np.zeros((len(X_train), num_class*num_model))

    class_weights = get_class_weights(y_train)

    for idx, model_name in enumerate(["lgb", "xgb", "cat"]):
        cv_scores = []
        cv_models = []
        val_pred_prob = np.zeros((len(X_train), 4))
        model = None
        for fold_idx, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
            X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[valid_index], y_train.iloc[train_index], \
                                       y_train.iloc[valid_index]
            logger.info(f"*****************{model_name} model {fold_idx + 1}/5 fold start training ***************")
            if model_name == "lgb":
                model, val_pred_prob = lgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
            elif model_name == "xgb":
                model, val_pred_prob = xgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
            elif model_name == "cat":
                model, val_pred_prob = cat_model_train(X_tr, y_tr, X_val, y_val, class_weights)

            score = macro_f1(y_val, val_pred_prob.argmax(1))
            logger.info(f"score = {score}")

            cv_scores.append(score)
            cv_models.append(model)
            train_oof_predicts[valid_index, idx*num_class:(1+idx)*num_class] = val_pred_prob
        logger.info("保存基础模型")
        dump_model(cv_models, model_name)

        logger.info(f"{model_name}_cv_scores:{cv_scores}")
        logger.info(f"{model_name}_cv_scores_mean: {np.mean(cv_scores)}")
        logger.info(f"{model_name}_cv_scores_std: {np.std(cv_scores)}")
        logger.info(f"{model_name}_oof_scores: {macro_f1(y_train, train_oof_predicts[:, idx*num_class:(1+idx)*num_class].argmax(1))}")

    logger.info(f"*****************final model start training ***************")
    # stacker
    X_train = pd.concat([X_train, pd.DataFrame(train_oof_predicts)], axis=1)
    final_cv_scores = []
    final_cv_models = []
    final_oof_preds = np.zeros((len(y_train), num_class))
    for fold_idx, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[valid_index], y_train.iloc[train_index], \
                                   y_train.iloc[valid_index]

        model, val_pred_prob = lgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        score = macro_f1(y_val, val_pred_prob.argmax(1))
        logger.info(f"{fold_idx+1}/5 score = {score}")

        final_cv_scores.append(score)
        final_cv_models.append(model)

        final_oof_preds[valid_index] = val_pred_prob
    logger.info("保存stacker模型")
    dump_model(final_cv_models, "stacker")

    logger.info(f"stacker_cv_scores:{final_cv_scores}")
    logger.info(f"stacker_cv_scores_mean: {np.mean(final_cv_scores)}")
    logger.info(f"stacker_cv_scores_std: {np.std(final_cv_scores)}")
    logger.info(f"stacker_oof_scores: {macro_f1(y_train, final_oof_preds.argmax(1))}")






























class Stacking(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res