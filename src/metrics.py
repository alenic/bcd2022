from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def pf1(y_true, y_prob):
    beta=1
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(y_true)):
        prediction = min(max(y_prob[idx], 0), 1)
        if (y_true[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def f1(y_true, y_pred):
    return f1_score(y_true.flatten(), y_pred.flatten())