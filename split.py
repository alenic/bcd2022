import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
import seaborn as sns

df = pd.read_csv(os.path.join("data","train.csv"))

y = df["cancer"].values
groups = df["patient_id"].values

cv = StratifiedGroupKFold(shuffle=True, random_state=42)
df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(df)), y, groups)):
    df.loc[val_idx, "fold"] = fold

    df_train = df[(df["fold"]!=fold)]
    df_val = df[(df["fold"]==fold)]

    # Check possible overlaps
    print(f"Fold Train {fold}, positives:", len(df_train[df_train["cancer"]==1]))
    print(f"Fold Val {fold}, positives:", len(df_val[df_val["cancer"]==1]))
    uniq_train = set(df_train["patient_id"].unique())
    uniq_val = set(df_val["patient_id"].unique())
    print("Overlap", uniq_train.intersection(uniq_val))

df.to_csv(os.path.join("data", "train_5fold.csv"),  index=False)
