import xgboost as xgb
from model.utils import macro_f1

def macro_f1_eval_metrics_xgb(preds, train_ds):
    lables = train_ds.get_label()
    preds = preds.reshape(-1, 4, order="F")
    preds = preds.argmax(1)

    return "macro_f1", -macro_f1(lables, preds)

def xgb_model_train(X_tr, y_tr, X_val, y_val, class_weights):
    y_tr_weight = y_tr.map(class_weights)
    train_ds = xgb.DMatrix(data=X_tr, label=y_tr, weight=y_tr_weight)
    valid_ds = xgb.DMatrix(data=X_val, label=y_val)
    # test_ds = xgb.DMatrix(data=X_test)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',  # multi:',
        'num_class': 4,
        'gamma': 1,
        'min_child_weight': 1.5,
        'max_depth': 5,
        'reg_lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'eta': 0.1,
        'random_state': 2022
    }

    watchlist = [(train_ds, 'train'), (valid_ds, 'eval')]

    model = xgb.train(params, train_ds, num_boost_round=1000, evals=watchlist, verbose_eval=100,
                      early_stopping_rounds=50, feval=macro_f1_eval_metrics_xgb)
    val_pred_prob = model.predict(valid_ds, ntree_limit=model.best_ntree_limit)
    # test_pred = model.predict(test_ds, ntree_limit=model.best_ntree_limit)

    return model, val_pred_prob
