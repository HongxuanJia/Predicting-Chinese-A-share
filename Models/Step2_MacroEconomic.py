import pandas as pd
import os
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.2f' % x)

def data(DirectorPath):
    director = os.listdir(DirectorPath)
    dflist = []
    for file in director:
        filepath = DirectorPath +'/' +file
        df = pd.read_excel(filepath)
        dflist.append(df)
    df_merged = dflist[0]['Accper']
    for df in dflist:
        df_merged = pd.merge(df_merged,df,how="outer")
    print(df.dtypes)
    print(df.shape)
    return df_merged


if __name__ == '__main__':
    output = data('./Macroeconomic Data')
    output.to_csv('./Macroeconomic Data/macroeconomic_features.csv')