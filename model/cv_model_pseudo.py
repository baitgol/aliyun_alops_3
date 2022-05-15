import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostClassifier
from model.utils import macro_f1, dump_model, load_model, get_class_weights
from feature.gen_features_v2 import cate_features
from config import project_path, model_path, logger
from model.lgb_model import lgb_model_train
from model.xgb_model import xgb_model_train
from model.cat_model import cat_model_train

def soft_max(x):
    x = np.exp(x)
    return x / np.expand_dims(np.sum(x, 1), 1)

def cv_model_train(X_train_, y_train_, unlabeled_train_, model_name="lgb"):
    unlabeled_train = deepcopy(unlabeled_train_)
    X_train = deepcopy(X_train_)
    y_train = deepcopy(y_train_)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2010)
    oof_preds = np.zeros((len(X_train), 4))
    val_pred_prob = np.zeros_like(oof_preds)
    cv_models = []
    model = None
    cv_scores = []
    class_weights = get_class_weights(y_train)

    logger.info(f"{'*' * 20}训练模型, model_name = {model_name} {'*' * 20}")

    logger.info(f"先预测伪标签")
    unlabeled_preds = np.zeros((len(unlabeled_train), 4))
    for idx, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[valid_index], y_train.iloc[train_index], \
                                   y_train.iloc[valid_index]


        # if model_name == "xgb" or model_name == "cat":
        logger.info(f"*****************{'lgb'} model {idx + 1}/5 fold start training ***************")
        model, val_pred_prob = lgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        unlabeled_pred = model.predict(unlabeled_train, num_iteration=model.best_iteration)
        # elif model_name == "lgb":
        #     logger.info(f"*****************{'xgb'} model {idx + 1}/5 fold start training ***************")
        #     model, val_pred_prob = xgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        #     unlabeled_ds = xgb.DMatrix(data=unlabeled_train)
        #     unlabeled_pred = model.predict(unlabeled_ds, ntree_limit=model.best_ntree_limit)

        # if model_name == "lgb" :
        #     logger.info(f"*****************{'lgb'} model {idx + 1}/5 fold start training ***************")
        #     model, val_pred_prob = lgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        #     unlabeled_pred = model.predict(unlabeled_train, num_iteration=model.best_iteration)
        # elif model_name == "xgb":
        #     logger.info(f"*****************{'xgb'} model {idx + 1}/5 fold start training ***************")
        #     model, val_pred_prob = xgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        #     unlabeled_ds = xgb.DMatrix(data=unlabeled_train)
        #     unlabeled_pred = model.predict(unlabeled_ds, ntree_limit=model.best_ntree_limit)
        # elif model_name == "cat":
        #     logger.info(f"*****************{'cat'} model {idx + 1}/5 fold start training ***************")
        #     model, val_pred_prob = cat_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        #     unlabeled_ds = Pool(data=unlabeled_train)
        #     unlabeled_pred = model.predict_proba(unlabeled_ds)

        val_pred = val_pred_prob.argmax(1)
        score = macro_f1(y_val, val_pred)
        logger.info(f"score: {score}")
        oof_preds[valid_index] = val_pred_prob
        # test_preds += test_pred / kfold.n_splits
        cv_scores.append(score)
        cv_models.append(model)
        unlabeled_preds += unlabeled_pred / 4

    logger.info(f"{model_name}_cv_scores:{cv_scores}")
    logger.info(f"{model_name}_cv_scores_mean: {np.mean(cv_scores)}")
    logger.info(f"{model_name}_cv_scores_std: {np.std(cv_scores)}")
    logger.info(f"{model_name}_oof_score: {macro_f1(y_train, oof_preds.argmax(1))}")
    #
    print("组装数据")
    unlabeled_train["label"] = unlabeled_preds.argmax(1)

    unlabeled_train = pd.concat([unlabeled_train,
                                 pd.DataFrame(unlabeled_preds, columns=[f"unlabel_{str(i)}" for i in range(4)])], axis=1)
    print(unlabeled_train.shape)
    # unlabeled_train = unlabeled_train.loc[unlabeled_train[f"unlabel_{str(unlabeled_train['label'])}"] >= 0.8]

    unlabeled_train = unlabeled_train.loc[((unlabeled_train["label"] == 0) & (unlabeled_train["unlabel_0"] >= 0.8))
                                          | ((unlabeled_train["label"] == 1) & (unlabeled_train["unlabel_1"] >= 0.8))
                                          # | ((unlabeled_train["label"] == 2) & (unlabeled_train["unlabel_2"] >= 0.8))
                                          # | ((unlabeled_train["label"] == 3) & (unlabeled_train["unlabel_3"] >= 0.8))
                                          ]

    print(unlabeled_train.shape)
    unlabeled_train = unlabeled_train.drop([f"unlabel_{str(i)}" for i in range(4)], axis=1)

    y_train = pd.concat([y_train, unlabeled_train["label"]], ignore_index=True)
    unlabeled_train = unlabeled_train.drop(["label"], axis=1)
    unlabeled_train["tag"] = 0
    X_train["tag"] = 1
    X_train = pd.concat([X_train, unlabeled_train], ignore_index=True)

    unlabeled_index = X_train.loc[X_train.tag == 0].index
    X_train.drop("tag", axis=1, inplace=True)
    print(X_train.shape, y_train.shape)

    logger.info(f"{'*' * 20}训练第二层基础模型, model_name = {model_name} {'*' * 20}")
    oof_preds = np.zeros((len(X_train), 4))
    val_pred_prob = np.zeros_like(oof_preds)
    cv_models = []
    model = None
    cv_scores = []
    class_weights = get_class_weights(y_train)

    for idx, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[valid_index], y_train.iloc[train_index], \
                                   y_train.iloc[valid_index]

        logger.info(f"*****************{model_name} model {idx + 1}/5 fold start training ***************")
        if model_name == "lgb":
            model, val_pred_prob = lgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        elif model_name == "xgb":
            model, val_pred_prob = xgb_model_train(X_tr, y_tr, X_val, y_val, class_weights)
        elif model_name == "cat":
            model, val_pred_prob = cat_model_train(X_tr, y_tr, X_val, y_val, class_weights)

        val_pred = val_pred_prob.argmax(1)
        score = macro_f1(y_val, val_pred)
        logger.info(f"score: {score}")

        oof_preds[valid_index] = val_pred_prob
        # test_preds += test_pred / kfold.n_splits

        cv_scores.append(score)
        cv_models.append(model)

    logger.info("保存模型")
    dump_model(cv_models, model_name)

    logger.info(f"{model_name}_cv_scores:{cv_scores}")
    logger.info(f"{model_name}_cv_scores_mean: {np.mean(cv_scores)}")
    logger.info(f"{model_name}_cv_scores_std: {np.std(cv_scores)}")

    oof_preds = pd.DataFrame(oof_preds).iloc[lambda x: ~x.index.isin(unlabeled_index)].values
    y_train = y_train.iloc[lambda x: ~x.index.isin(unlabeled_index)]

    logger.info(f"{model_name}_oof_score: {macro_f1(y_train, oof_preds.argmax(1))}")

    return oof_preds



def cv_model_eval(X_train, y_train, model_name="lgb"):

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2010)
    eval_preds = np.zeros((len(X_train), 4))
    eval_pred = np.zeros_like(eval_preds)

    num_model = 5
    models = load_model(model_name, num_model)
    logger.info(f"{'*' * 20}, model_name = {model_name} {'*' * 20}")
    cv_scores = []
    for idx, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[valid_index], y_train.iloc[train_index], \
                                   y_train.iloc[valid_index]

        model = models[idx]
        logger.info(f"*****************{model_name} model {idx + 1}/5 fold start training ***************")
        if model_name == "lgb":
            eval_pred = model.predict(X_val, num_iteration=model.best_iteration)
        elif model_name == "xgb":
            eval_ds = xgb.DMatrix(data=X_val)
            eval_pred = model.predict(eval_ds, ntree_limit=model.best_ntree_limit)
        elif model_name == "cat":
            eval_ds = Pool(data=X_val, cat_features=cate_features)
            eval_pred = model.predict_proba(eval_ds)

        score = macro_f1(y_val, eval_pred.argmax(1))
        logger.info(f"score: {score}")

        eval_preds[valid_index] = eval_pred
        # test_preds += test_pred / kfold.n_splits
        cv_scores.append(score)

    logger.info(f"{model_name}_cv_scores:{cv_scores}")
    logger.info(f"{model_name}_cv_scores_mean: {np.mean(cv_scores)}")
    logger.info(f"{model_name}_cv_scores_std: {np.std(cv_scores)}")
    logger.info(f"{model_name}_oof_score: {macro_f1(y_train, eval_preds.argmax(1))}")
    return eval_preds


def cv_model_predict(X_test, model_name="lgb"):
    logger.info(f"{'*' * 20}模型预测, model_name = {model_name} {'*' * 20}")
    test_preds = np.zeros((len(X_test), 4))
    test_pred = np.zeros_like(test_preds)

    num_model = 5
    models = load_model(model_name, num_model)

    for idx, model in enumerate(models):
        logger.info(f"*****************{model_name} model {idx + 1}/5 fold start training ***************")
        if model_name == "lgb":
            # test_ds = lgb.Dataset(data=X_test)
            test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        elif model_name == "xgb":
            test_ds = xgb.DMatrix(data=X_test)
            test_pred = model.predict(test_ds, ntree_limit=model.best_ntree_limit)
        elif model_name == "cat":
            test_ds = Pool(data=X_test)#, cat_features=cate_features)
            test_pred = model.predict_proba(test_ds)

        test_preds += test_pred / num_model

    return test_preds


