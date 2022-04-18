import pandas as pd

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
        macro_F1 += weights[i] * F1
    return macro_F1