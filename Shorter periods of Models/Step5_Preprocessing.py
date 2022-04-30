import pandas as pd
import numpy as np

import miceforest as mf
import matplotlib.pyplot as plt
import missingno as msno
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.2f' % x)

def PreprocessingData():
    db = missing_info()['db']
    missing_columns = missing_info()['missing_colmns']
    # preprocessing
    db = db.drop(columns=missing_columns,axis=1)
    try:
        db_amp = mf.ampute_data(db,perc=0.9,random_state=10)
        kernel = mf.ImputationKernel(
            db_amp,
            datasets=4,
            save_all_iterations=True,
            random_state=1
        )
        kernel.mice(2)
        db = kernel.complete_data(dataset=0,inplace=False)
    except ValueError as e:
        print(e.args)
    db.to_csv('./datasets/dataset.csv')
    missing = db.isnull().sum().reset_index().rename(columns={0: 'missNum'})
    missing['missRate'] = missing['missNum'] / db.shape[0]
    print('Missing rate after preprocessing:\n{}'.format(missing.head(500)))
    print(db.dtypes)
    print('The count of total database:\n{}\n-------------------------------------'.format(db.count()))
    print('The description of total database:\n{}\n-------------------------------------'.format(db.describe()))
    msno.matrix(db,filter=None, n=0, p=0, sort=None, figsize=(25, 10),
                width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
                fontsize=16, labels=None, sparkline=True, freq=None, ax=None)
    plt.show()

def numeric(path1,path2):
    db = pd.read_csv(path1)
    del db['Stknme']
    columns_list = db.columns.values
    columns_list = columns_list.tolist()
    columns_list.remove('Stkcd')
    columns_list.remove('Accper')
    return_list = ['abnormal_return_1', 'abnormal_return_3', 'abnormal_return_5', 'abnormal_return_7']
    columns_list = list(set(columns_list)-set(return_list))
    try:
        for column in columns_list:
            db[column] = ProcessingNumericalData(db[column])
    except KeyError as e:
        print(e.args)
    db.to_csv(path2)

def missing_info():
    db = pd.read_csv('./Features&Targets/datasets.csv',low_memory=False)
    del db['Unnamed: 0.1']
    del db['Unnamed: 0']
    del db['Typrep']
    # del db['Stkmne']
    db['Stkcd'] = db['Stkcd'].astype(int)
    db = db.set_index(['Stkcd'])
    print('Missing rate before preprocessing:')
    missing = db.isnull().sum().reset_index().rename(columns={0: 'missNum'})
    missing['missRate'] = missing['missNum'] / db.shape[0]
    # print(missing.head(500))
    # print('-------------------------------------')
    missing.to_excel('./paper_figures/miss_info.xlsx')
    msno.matrix(db, filter=None, n=0, p=0, sort=None, figsize=(25, 10),
                width_ratios=(15, 1), color=(0.624, 0.502, 0.725),
                fontsize=16, labels=None, sparkline=True, freq=None, ax=None)
    # msno.heatmap(db)
    plt.savefig('./figures/missing_data.png', bbox_inches='tight', dpi=2400)
    plt.show()
    missing_columns = missing[(missing['missRate']>=0.5)]['index']
    missing_columns = missing_columns.tolist()
    # print(db.head(10))
    json={
        'missing_colmns': missing_columns,
        'db': db,
    }
    return json

def ProcessingNumericalData(input):
    mean = input.mean()
    std = input.std()
    bins = [(mean-3*std),(mean-2.5*std),(mean-2*std),(mean-1.5*std),(mean-std),(mean-0.5*std),
            mean,
            (mean+0.5*std),(mean+std),(mean+1.5*std),(mean+2*std), (mean+2.5*std),(mean+3*std)
            ]
    category = np.digitize(input,bins=bins)
    return category

def main():
    missing_info()
    # PreprocessingData()
    # FeatureSelection(PreprocessingData())
    # numeric('./datasets/new_dataset.csv','./datasets/numeric_dataset.csv')


if __name__ == '__main__':
    main()
    # missing_info()