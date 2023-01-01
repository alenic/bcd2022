import pandas as pd
from sklearn.preprocessing import LabelEncoder

def df_preprocess(df: pd.DataFrame, softlabel=False):
    COLS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
    new_df = df.copy()
    # Age
    new_df.age.fillna(new_df.age.mean(), inplace=True)

    new_df["target"] = new_df["cancer"]
    new_df['age'] = pd.qcut(new_df.age, 3, labels=range(3), retbins=False).astype(int)
    
    # Only two classes for BIRADS
    new_df.loc[new_df["BIRADS"]==0, "BIRADS"] = 1 # required a follow up
    new_df.loc[new_df["BIRADS"]>0, "BIRADS"] = 0 # normal breasts

    new_df.loc[(new_df["difficult_negative_case"]==1), "cancer"] = 0.14
    #new_df.loc[(new_df["biopsy"]==1) & (new_df["target"]==0), "cancer"] = 0.28


    #dfg = new_df.groupby(["patient_id", "laterality"])["cancer"].count().reset_index()
    #new_df = pd.merge(new_df, dfg, on=["patient_id", "laterality"], ).rename(columns={"cancer_x": "cancer", "cancer_y": "count_pl"})
    #new_df["cancer"] = new_df["cancer"]/new_df["count_pl"]

    new_df.loc[:, COLS] = new_df[COLS].apply(LabelEncoder().fit_transform)

    # 100 is the label to ignore
    new_df.loc[df["density"].isna(), "density"] = 100

    return new_df