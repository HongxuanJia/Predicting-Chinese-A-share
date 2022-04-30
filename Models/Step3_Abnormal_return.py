import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.4f' % x)

# def ReturnData(filePath):
#     rf = pd.read_csv(filePath,low_memory=False)
#     rf = rf[['Stkcd','Trddt','Stknme','Nindnme','Nshripo','Dnshrtrd','Dnvaltrd','Dretnd','Cdretmdeq']]
#     rf.to_csv('./Stock Return/A_return.csv')

def abnormal_15(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd','Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_15 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_15 = pd.DataFrame(abnormal_15)
    abnormal_15['Trddt']=start
    abnormal_15.rename(columns={"daily_return": "abnormal_return_15"}, inplace=True)
    return abnormal_15

def abnormal_30(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_30 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_30 = pd.DataFrame(abnormal_30)
    abnormal_30['Trddt'] = start
    abnormal_30.rename(columns={"daily_return": "abnormal_return_30"}, inplace=True)
    return abnormal_30

def abnormal_45(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_45 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_45 = pd.DataFrame(abnormal_45)
    abnormal_45['Trddt'] = start
    abnormal_45.rename(columns={"daily_return": "abnormal_return_45"}, inplace=True)
    return abnormal_45

def abnormal_60(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_60 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_60 = pd.DataFrame(abnormal_60)
    abnormal_60['Trddt'] = start
    abnormal_60.rename(columns={"daily_return": "abnormal_return_60"}, inplace=True)
    return abnormal_60

def abnormal_75(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_75 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_75 = pd.DataFrame(abnormal_75)
    abnormal_75['Trddt'] = start
    abnormal_75.rename(columns={"daily_return": "abnormal_return_75"}, inplace=True)
    return abnormal_75

def abnormal_90(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_90 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_90 = pd.DataFrame(abnormal_90)
    abnormal_90['Trddt'] = start
    abnormal_90.rename(columns={"daily_return": "abnormal_return_90"}, inplace=True)
    return abnormal_90

def return_15():
    list = []
    for i in range(0,20):
        Q1 = abnormal_15(20000331+i*10000,20000415+i*10000)
        Q2 = abnormal_15(20000630+i*10000,20000715+i*10000)
        Q3 = abnormal_15(20000930+i*10000, 20001015+i*10000)
        Q4 = abnormal_15(20001231+i*10000, 20000115+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_15 = pd.concat(list,axis=0)
    return data_15

def return_30():
    list = []
    for i in range(0,20):
        Q1 = abnormal_30(20000331+i*10000,20000430+i*10000)
        Q2 = abnormal_30(20000630+i*10000,20000731+i*10000)
        Q3 = abnormal_30(20000930+i*10000, 20001031+i*10000)
        Q4 = abnormal_30(20001231+i*10000, 20000131+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_30 = pd.concat(list,axis=0)
    return data_30

def return_45():
    list = []
    for i in range(0,20):
        Q1 = abnormal_45(20000331+i*10000,20000515+i*10000)
        Q2 = abnormal_45(20000630+i*10000,20000815+i*10000)
        Q3 = abnormal_45(20000930+i*10000, 20001115+i*10000)
        Q4 = abnormal_45(20001231+i*10000, 20000215+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_45 = pd.concat(list,axis=0)
    return data_45

def return_60():
    list = []
    for i in range(0,20):
        Q1 = abnormal_60(20000331+i*10000,20000531+i*10000)
        Q2 = abnormal_60(20000630+i*10000,20000815+i*10000)
        Q3 = abnormal_60(20000930+i*10000, 20001130+i*10000)
        Q4 = abnormal_60(20001231+i*10000, 20000219+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_60 = pd.concat(list,axis=0)
    return data_60

def return_75():
    list = []
    for i in range(0,20):
        Q1 = abnormal_75(20000331+i*10000,20000615+i*10000)
        Q2 = abnormal_75(20000630+i*10000,20000915+i*10000)
        Q3 = abnormal_75(20000930+i*10000, 20001215+i*10000)
        Q4 = abnormal_75(20001231+i*10000, 20000315+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_75 = pd.concat(list,axis=0)
    return data_75

def return_90():
    list = []
    for i in range(0,20):
        Q1 = abnormal_90(20000331+i*10000,20000630+i*10000)
        Q2 = abnormal_90(20000630+i*10000,20000930+i*10000)
        Q3 = abnormal_90(20000930+i*10000, 20001231+i*10000)
        Q4 = abnormal_90(20001231+i*10000, 20000331+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_90 = pd.concat(list,axis=0)
    return data_90

if __name__ == '__main__':
    data_15 = return_15()
    data_30 = return_30()
    data_45 = return_45()
    data_60 = return_60()
    data_75 = return_75()
    data_90 = return_90()
    targets = data_15.merge(data_30)
    targets = pd.merge(targets,data_45)
    targets = pd.merge(targets, data_60)
    targets = pd.merge(targets, data_75)
    targets = pd.merge(targets, data_90)
    targets.to_csv('./Stock Return/targets.csv')



