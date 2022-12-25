import pandas as pd
from sklearn.preprocessing import LabelEncoder

def df_preprocess(df: pd.DataFrame):
    COLS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
    new_df = df.copy()
    new_df.age.fillna(new_df.age.mean(), inplace=True)
    new_df['age'] = pd.qcut(new_df.age, 10, labels=range(10), retbins=False).astype(int)
    
    new_df[COLS] = new_df[COLS].apply(LabelEncoder().fit_transform)

    return new_df