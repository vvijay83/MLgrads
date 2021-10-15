import pandas as pd
from includes import *
class Utilities(object):
     def __init__(self):
         print("Init")

     def read_data_to_df(file_name_loc):
            return (pd.read_csv(file_name_loc, sep='\t', parse_dates=[2]))

     def check_if_missing_val_imputation_needed(in_df: pd.DataFrame):
         # check for the missing values
         list_missing_val = in_df.isna().sum().sum()
         if list_missing_val > 0:
             return True
         return False


     def isNaN(num):
         return num != num

     def get_diff_between_dates(df, index, offset):
         d1 = df.loc[index, 'datetime']
         d2 = df.loc[index + offset, 'datetime']
         return (d2 - d1)

     def getColslistneedingImputation(df: pd.DataFrame):
        list_feat_to_impute = []
        cols_to_impute = df.columns
        for col in cols_to_impute:
            if df[col].isna().sum() != 0:
                list_feat_to_impute.append(col)
        return list_feat_to_impute

     def normalize_num_features(num_df, num_cols):
         scaler = MinMaxScaler().fit(num_df[num_cols])
         num_df[num_cols] = scaler.transform(num_df[num_cols])
         return num_df[num_cols]