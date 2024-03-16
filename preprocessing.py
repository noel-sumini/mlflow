import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, LabelEncoder

def load_csv(path):
    df = pd.read_csv(path)
    return df

def split_data(df, test_size = 0.2, random_state = 2024):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def binary_encoding(df, cols):
    binarizer = LabelBinarizer()
    binarizer.fit(df.loc[:, cols])
    df.loc[:, cols] =  binarizer.transform(df.loc[:, cols].copy())
    return df

def outlier_smoothing(df):
    for col in df.describe().columns:
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25) 
        df.loc[df[col] > df[col].quantile(0.75) + iqr * 1.5, col] = df[col].quantile(0.75)
    return df


def scaling(df, label_col):
    nuemric_cols = list(set(df.describe().columns) - set(label_col))
    scaler = MinMaxScaler()
    scaler.fit(df.loc[:, nuemric_cols ])
    df.loc[:, nuemric_cols] =  scaler.transform(df.loc[:, nuemric_cols])
    return df

def feature_selection(df):
    corr_df = df.corr().iloc[:, 0].apply(lambda x: np.abs(x))
    selected_cols = corr_df.loc[corr_df > 1 / df.shape[1]].index
    return selected_cols

def split_data_label(df, label_col):
    cols = list(df.columns)
    if label_col in cols:
        cols.remove(label_col)
    data = df.loc[:, cols]
    label = df.loc[:, label_col].astype("int")
    return data, label

def preprocessing(df, train = True):
    if 'id' in df.columns:
        df.drop(columns = ['id'], inplace = True)

    label_col = ['diagnosis']

    if train:
        train_data, test_data = split_data(df)
    else:
        train_data = df.copy()
        test_data = df.copy()

    train_data = binary_encoding(train_data, label_col)
    test_data = binary_encoding(test_data, label_col)
    
    train_data = outlier_smoothing(train_data )
    test_data = outlier_smoothing(test_data)

    train_data = scaling(train_data,  label_col)
    test_data = scaling(test_data, label_col)
    

    nuemric_cols = list( set(df.describe().columns) - set(label_col) )
    selected_cols = feature_selection(train_data.loc[:, nuemric_cols])
    train_df = train_data.loc[:, list(selected_cols) + label_col ].copy()
    test_df = test_data.loc[:, list(selected_cols) + label_col ].copy()

    return train_df, test_df



    
    

    




