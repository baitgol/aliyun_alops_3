import numpy as np
import lightgbm as lgb
from scipy.misc import derivative

from model.utils import macro_f1

def soft_max(x):
    x = np.exp(x)
    return x / np.expand_dims(np.sum(x, 1), 1)



def log_loss_lgb(preds, train_ds):

    num_class = 4

    lables = train_ds.get_label()
    lables = np.eye(num_class)[lables.astype('int')]
    preds = preds.reshape(-1, num_class, order='F')
    def fl(x, t):
        p = soft_max(x)
        return -t * np.log(p)
    partial_fl = lambda x: fl(x, lables)
    grad = derivative(partial_fl, preds, n=1, dx=1e-6)
    hess = derivative(partial_fl, preds, n=2, dx=1e-6)

    # print(hess)
    return grad.flatten('F'), hess.flatten('F')

def macro_f1_eval_metrics_lgb(preds, train_ds):
    lables = train_ds.get_label()

    preds = preds.reshape(-1, 4, order="F")
    preds = preds.argmax(1)

    return "macro_f1", macro_f1(lables, preds), True




def lgb_model_train(X_tr, y_tr, X_val, y_val, class_weights):
    y_tr_weight = y_tr.map(class_weights)
    train_ds = lgb.Dataset(data=X_tr, label=y_tr, weight=y_tr_weight)
    valid_ds = lgb.Dataset(data=X_val, label=y_val, reference=train_ds)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        ' min_child_samples': 20,
        'num_leaves': 63,
        'lambda_l1': 1,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'seed': 2022,
        'n_jobs': -1,
        'verbose': -1
    }
    model = lgb.train(params, train_ds, num_boost_round=1000, valid_sets=[train_ds, valid_ds],
                      categorical_feature=[],
                      verbose_eval=100, early_stopping_rounds=50, feval=macro_f1_eval_metrics_lgb
                      )
    val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)

    return model, val_pred_prob