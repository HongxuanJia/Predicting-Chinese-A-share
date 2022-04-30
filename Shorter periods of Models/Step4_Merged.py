import pandas as pd
import numpy as np
from Step1_Financial import DataPreprocessing
from Step2_MacroEconomic import data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format',lambda x: '%.2f' % x)

def Features():
    fd = pd.read_csv('./Financial Data/financial_features.csv')
    # md = pd.read_csv('./Macroeconomic Data/macroeconomic_features.csv')
    print(fd.shape)
    # print(md.shape)
    # features_dataset = pd.merge(fd,md,on=['Accper'],how='outer')
    fd.to_csv('./Features&Targets/features.csv')
    print(fd.shape)
    # print(features_dataset.head(100),features_dataset.shape)
    # return features_dataset

def dataset():
    # Features_Merged()
    features = pd.read_csv('./Features&Targets/features.csv')
    targets = pd.read_csv('./Stock Return/targets.csv')
    del features['Unnamed: 0']
    del targets['Unnamed: 0']
    print(features.shape)
    print(targets.shape)
    targets.rename(columns={'Trddt' : 'Accper','abnormal_return_15':'abnormal_return_1','abnormal_return_30':'abnormal_return_3','abnormal_return_45':'abnormal_return_5','abnormal_return_60':'abnormal_return_7'}, inplace=True)
    database = pd.concat([features,targets],axis=1)
    print(database.shape)
    database.to_csv('./Features&Targets/datasets.csv')

def main():
    features = pd.read_csv('./datasets/dataset.csv')
    targets = pd.read_csv('./Stock Return/targets.csv')
    targets.rename(columns={'Trddt' : 'Accper','abnormal_return_15':'abnormal_return_1','abnormal_return_30':'abnormal_return_3','abnormal_return_45':'abnormal_return_5','abnormal_return_60':'abnormal_return_7'}, inplace=True)
    database = pd.merge(features, targets)
    database.to_csv('./datasets/new_dataset.csv')

if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
