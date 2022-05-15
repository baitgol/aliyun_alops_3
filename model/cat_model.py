import numpy as np
from catboost import Pool, CatBoostClassifier
from model.utils import macro_f1

from feature.gen_features_v2 import cate_features


# class LoglossObjective(object):
#     def calc_ders_range(self, approxes, targets, weights):
#         assert len(approxes) == len(targets)
#         if weights is not None:
#             assert len(weights) == len(approxes)
#
#         result = []
#         for index in range(len(targets)):
#             e = np.exp(approxes[index])
#             p = e / (1 + e)
#             der1 = targets[index] - p
#             der2 = -p * (1 - p)
#
#             if weights is not None:
#                 der1 *= weights[index]
#                 der2 *= weights[index]
#
#             result.append((der1, der2))
#         return result

#
# model = CatBoostClassifier(loss_function=LoglossObjective())

class CatBoostMetric(object):
    # def get_final_error(self, error):
    #     return error / (weight + 1e-38)#可以自己定义error_sum与权重的关系

    def is_max_optimal(self):
        return True

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def evaluate(self, approxes, target, weight):
        pred = np.array(approxes)
        pred = pred.argmax(0)
        f1_sum = macro_f1(target, pred)
        return f1_sum, 1



def cat_model_train(X_tr, y_tr, X_val, y_val, class_weights):
    params = {

        'classes_count': 4,
        'learning_rate': 0.1,
        'depth': 6,
        'class_weights': class_weights,
        'reg_lambda': 10,
        'min_data_in_leaf': 20,
        # 'subsample': 0.8,
        # 'colsample_bylevel': 0.8,
        'use_best_model': True,
        'loss_function': 'MultiClass',
         # 'eval_metric': CatBoostMetric(), # , 'MultiClass',
        'random_state': 2022}

    train_ds = Pool(data=X_tr, label=y_tr)#, cat_features=cate_features)
    val_ds = Pool(data=X_val, label=y_val)#, cat_features=cate_features)

    model = CatBoostClassifier(iterations=1000, **params)
    model.fit(train_ds, eval_set=(val_ds), use_best_model=True, verbose=100, early_stopping_rounds=50)
    val_pred_prob = model.predict_proba(X_val)

    return model, val_pred_prob