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

def train_test_validate():
    db = pd.read_csv('./datasets/dataset.csv')
    columns_list = db.columns.values
    columns_list = columns_list.tolist()
    columns_list.remove('Stkcd')
    columns_list.remove('Accper')
    return_list = ['abnormal_return_15', 'abnormal_return_30', 'abnormal_return_45', 'abnormal_return_60', 'abnormal_return_75', 'abnormal_return_90']
    columns_list = list(set(columns_list)-set(return_list))
    try:
        for column in columns_list:
            db[column] = ProcessingNumericalData(db[column])
    except KeyError as e:
        print(e.args)
    db.to_csv('./datasets/numeric_dataset.csv')

def missing_info():
    db = pd.read_csv('./Features&Targets/datasets.csv')
    del db['Unnamed: 0']
    del db['Unnamed: 0_x']
    del db['Unnamed: 0_y']
    del db['Stknme']
    del db['Typrep']
    del db['Investment Climate Index: Overall']
    db['Stkcd'] = db['Stkcd'].astype(int)
    db = db.set_index(['Stkcd'])
    db.rename(columns={'Industrial Enterprise: Interest Expense: YTD': 'Industrial Enterprise_Interest Expense',
                       'GDP: Current Prices': 'GDP', 'CPI: YoY': 'CPI',
                       'M2: YoY': 'M2',
                       'Industrial Enterprises: Total Profit: YTD': 'Industrial Enterprises_Total Profit',
                       'Value of Imports: RMB': 'Value of Imports',
                       'Industrial Enterprises: Main Business Income: YTD': 'Industrial Enterprises_Main Business Income',
                       'Value of Exports: RMB': 'Value of Exports',
                       'Yicai Research Institute: China Financial Condition Index (Daily)': 'Yicai Research Institute_China Financial Condition Index',
                       'Industrial Enterprises: Financial Expenses: YTD': 'Industrial Enterprises_Financial Expenses',
                       'Industrial Enterprise: Inventory: YTD': 'Industrial Enterprise_Inventory',
                       'Macro-economic Climate Index: Pre-warning Index': 'Macro-economic Climate Index_Pre-warning Index',
                       'Industrial Enterprises: Total Liabilities': 'Industrial Enterprises_Total Liabilities',
                       'Investment Climate Index: Overall': 'Investment Climate Index',
                       'Loan Prime Rate (LPR): 1Y': 'Loan Prime Rate (LPR)',
                       'China Bulk Commodity Price Index: General Index': 'Price Index',
                       'Industrial Enterprises: Main Business Cost: YTD': 'Industrial Enterprises_Main Business Cost',
                       'Industrial Enterprises: Total Assets': 'Industrial Enterprises_Total Assets',
                       'FAI: YTD': 'FAI',
                       'Industrial Enterprise: Accounts Receivable': 'Industrial Enterprise_Accounts Receivable',
                       'Consumer Confidence Index: Q': 'Consumer Confidence Index',
                       }, inplace=True)

    print('Missing rate before preprocessing:')
    missing = db.isnull().sum().reset_index().rename(columns={0: 'missNum'})
    missing['missRate'] = missing['missNum'] / db.shape[0]
    # print(missing.head(500))
    # print('-------------------------------------')
    missing.to_excel('./paper_figures/miss_info.xlsx')
    msno.matrix(db, filter=None, n=0, p=0, sort=None, figsize=(25, 10),
                width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
                fontsize=16, labels=None, sparkline=True, freq=None, ax=None)
    # msno.heatmap(db)
    plt.show()
    missing_columns = missing[(missing['missRate']>=0.5)]['index']
    missing_columns = missing_columns.tolist()
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
    # PreprocessingData()
    # FeatureSelection(PreprocessingData())
    train_test_validate()


if __name__ == '__main__':
    # main()
    missing_info()