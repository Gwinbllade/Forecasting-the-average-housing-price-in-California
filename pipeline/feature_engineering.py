import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
import columns 
import pickle
warnings.filterwarnings("ignore")

param_dict = pickle.load(open('pipeline/param_dict.pickle', 'rb'))

def impute_na(df, variable, value):
    return df[variable].fillna(value)



def encoding(ds):
    encoder = OneHotEncoder(categories='auto',
                        sparse_output=False,
                        drop='first', 
                        handle_unknown='error') 

    one_hot_cols = columns.cat_columns

    encoder.fit(ds[one_hot_cols])

    tmp = encoder.transform(ds[one_hot_cols])

    ohe_output = pd.DataFrame(tmp)
    ohe_output.columns = encoder.get_feature_names_out(one_hot_cols)
    ds = ds.drop(one_hot_cols, axis=1)
    ds = pd.concat([ds, ohe_output], axis=1)
    return ds

def outlieers_procesing(ds):
    for column  in columns.outliers_columns:
        Q1=ds[column].quantile(0.25)
        Q3=ds[column].quantile(0.75)
        IQR=Q3-Q1
        Lower_limit=Q1-1.5*IQR
        Upper_limit=Q3+1.5*IQR
        ds[column][ds[column]<Lower_limit] =Lower_limit
        ds[column][ds[column]>Upper_limit] =Upper_limit
    return ds

def mean_impute(ds):
    for column in columns.mean_impute_columns:
        ds[column] = impute_na(ds, column, param_dict['mean_impute_values'][column])
    return ds

def fr_engineering(ds):
    ds = ds.drop(columns = columns.drop_columns, axis=1)

    ds = mean_impute(ds)
    ds = encoding(ds)
    ds = outlieers_procesing(ds)
    
    

    return ds