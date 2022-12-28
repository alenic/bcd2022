import pandas as pd
from sklearn.preprocessing import LabelEncoder

def df_preprocess(df: pd.DataFrame, softlabel=False):
    COLS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
    new_df = df.copy()
    # Age
    new_df.age.fillna(new_df.age.mean(), inplace=True)
    new_df['age'] = pd.qcut(new_df.age, 10, labels=range(10), retbins=False).astype(int)
    
    if softlabel:
        new_df["target"] = new_df["cancer"]
        new_df.loc[new_df["difficult_negative_case"]==1, "cancer"] = 0.5

    new_df[COLS] = new_df[COLS].apply(LabelEncoder().fit_transform)

    # Use difficult_negative_case
    
    return new_df