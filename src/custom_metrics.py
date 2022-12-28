from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

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
    c_recall = 0 if y_true_count==0 else ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def f1(y_true, y_pred, average="binary"):
    return pf1(y_true.flatten(), y_pred.flatten(), average=average)


def grouped_mean(y_true, y_prob, df_val, thr):
    df = pd.DataFrame()
    df["pid"] = df_val["patient_id"].astype(str) + "_" + df_val["laterality"].astype(str)
    df["y_prob"] = y_prob
    df["y_true"] = y_true
    df_true = df.groupby("pid")["y_true"].apply(lambda x: x.sum()>0).astype(int)
    df_pred = df.groupby("pid")["y_prob"].mean()

    return pf1(df_true.values, (df_pred.values >= thr).astype(int))


def grouped_reduced(y_true, y_pred, df_val, reduce="max"):
    df = pd.DataFrame()
    df["pid"] = df_val["patient_id"].astype(str) + "_" + df_val["laterality"].astype(str)
    df["y_pred"] = y_pred
    df["y_true"] = y_true
    df_true = df.groupby("pid")["y_true"].apply(lambda x: x.sum()>0).astype(int)
    if reduce == "max":
        df_pred = df.groupby("pid")["y_pred"].max()
    elif reduce == "majority":
        df_pred = df.groupby("pid")["y_pred"].apply(lambda x: x.sum() >= len(x)*0.5).astype(int)

    return pf1(df_true.values, df_pred.values)

