import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.2f' % x)

def DataPreprocessing(BalanceSheet_file, CashFlow_file, IncomeStatement_File):
    BalanceSheet = pd.read_csv(BalanceSheet_file)
    BalanceSheet = BalanceSheet.dropna(axis=1, how="all")
    # print(BalanceSheet.head(10))
    # print(BalanceSheet.shape)
    # print(BalanceSheet.count(),BalanceSheet.shape,BalanceSheet.describe(),BalanceSheet.dtypes)
    CashFlow = pd.read_csv(CashFlow_file)
    CashFlow = CashFlow.dropna(axis=1,how="all")
    # print(CashFlow.head(10))
    # print(CashFlow.shape)
    IncomeStatement = pd.read_csv(IncomeStatement_File)
    IncomeStatement = IncomeStatement.dropna(axis=1, how="all")
    # print(IncomeStatement.head(10))
    # print(IncomeStatement.shape)
    fd_0 = pd.merge(BalanceSheet,CashFlow,how="outer")
    fd = pd.merge(fd_0,IncomeStatement,how="outer")
    fd = fd[~fd['Typrep'].isin(['B'])]
    fd = fd[~fd['Accper'].isin([20000101,20010101,20020101,20030101,
                                20040101,20050101,20060101,20070101,
                                20080101,20090101,20100101,20110101,
                                20120101,20130101,20140101,20150101,
                                20160101,20170101,20180101,20190101,20200101])]
    fd.set_index(['Stkcd','Accper'])
    print(fd.head(10))
    print(fd.dtypes)
    print(fd.shape)
    return fd

# def MacroPreprocessing(director):
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ffd = DataPreprocessing('./Financial Data/balance_sheet.csv','./Financial Data/cash_flow_direct.csv','./Financial Data/income.csv')
    ffd.to_csv('./Financial Data/financial_features.csv')