import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

root = os.path.join(os.environ["DATASET_ROOT"], "Breast-Cancer-Detection-2022")
print(root)

df = pd.read_csv(os.path.join(root,"train.csv"))

df_train, df_val = train_test_split(df, test_size=0.25, stratify=df.cancer, shuffle=True, random_state=42)

df_train.to_csv(os.path.join(root, "train_split.csv"),  index=False)
df_val.to_csv(os.path.join(root, "val_split.csv"),  index=False)