import pandas as pd
import numpy as np

def clean_outliers(df):
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        
        df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
    return df

def feature_engineering(df):
    
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    return df