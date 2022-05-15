import os
import pickle
import numpy as np
import pandas as pd
from scipy.misc import derivative
from sklearn.utils.class_weight import compute_class_weight

from config import model_path

def get_class_weights(train_label):
    classes = np.unique(train_label)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_label)
    class_weights = dict(zip(classes, weights))
    return class_weights


def dump_model(models, model_name):
    for idx, model in enumerate(models):
        model_file = os.path.join(model_path, f"{model_name}_{idx}.dat")
        pickle.dump(model, open(model_file, "wb"))


def load_model(model_name, num_model):
    models = []
    for idx in range(num_model):
        model_file = os.path.join(model_path, f"{model_name}_{idx}.dat")
        loaded_model = pickle.load(open(model_file, "rb"))
        models.append(loaded_model)
    return models



def macro_f1(target, pred) -> float:
    # weights = [3 / 7, 2 / 7, 1 / 7, 1 / 7]
    weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]
    df = pd.DataFrame({"target": target, "predict": pred})

    macro_F1 = 0.
    for i in range(len(weights)):
        TP = len(df[(df['target'] == i) & (df['predict'] == i)])
        FP = len(df[(df['target'] != i) & (df['predict'] == i)])
        FN = len(df[(df['target'] == i) & (df['predict'] != i)])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # print(precision, recall, F1)
        macro_F1 += weights[i] * F1
    return macro_F1



def focal_loss_lgb(y_true, y_pred):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    num_class = 4
    alpha = 0.75
    gamma = 2
    a, g = alpha, gamma

    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')


    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')


def focal_loss_lgb_eval_error(y_true, y_pred, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # a variant can be np.sum(loss)/num_class
    return 'focal_loss', np.mean(loss), False