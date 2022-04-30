import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.4f' % x)

# def ReturnData(filePath):
#     rf = pd.read_csv(filePath,low_memory=False)
#     rf = rf[['Stkcd','Trddt','Stknme','Nindnme','Nshripo','Dnshrtrd','Dnvaltrd','Dretnd','Cdretmdeq']]
#     rf.to_csv('./Stock Return/A_return.csv')

def abnormal_1(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd','Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_1 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_1 = pd.DataFrame(abnormal_1)
    abnormal_1['Trddt']=start
    abnormal_1.rename(columns={"daily_return": "abnormal_return_15"}, inplace=True)
    return abnormal_1

def abnormal_3(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_3 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_3 = pd.DataFrame(abnormal_3)
    abnormal_3['Trddt'] = start
    abnormal_3.rename(columns={"daily_return": "abnormal_return_30"}, inplace=True)
    return abnormal_3

def abnormal_5(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_5 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_5 = pd.DataFrame(abnormal_5)
    abnormal_5['Trddt'] = start
    abnormal_5.rename(columns={"daily_return": "abnormal_return_45"}, inplace=True)
    return abnormal_5

def abnormal_7(start,end):
    rf = pd.read_csv('./Stock Return/A_return.csv')
    rf = rf[(rf.Trddt>start) & (rf.Trddt<end)]
    del rf['Unnamed: 0']
    rf.set_index(['Stkcd', 'Trddt'])
    rf['daily_return'] = rf['Dretnd']-rf['Cdretmdeq']
    abnormal_7 = rf.groupby(by=['Stknme','Stkcd'])['daily_return'].sum().reset_index()
    abnormal_7 = pd.DataFrame(abnormal_7)
    abnormal_7['Trddt'] = start
    abnormal_7.rename(columns={"daily_return": "abnormal_return_60"}, inplace=True)
    return abnormal_7


def return_1():
    list = []
    for i in range(0,20):
        Q1 = abnormal_1(20000331+i*10000,20000402+i*10000)
        Q2 = abnormal_1(20000630+i*10000,20000701+i*10000)
        Q3 = abnormal_1(20000930+i*10000, 20001002+i*10000)
        Q4 = abnormal_1(20001231+i*10000, 20000102+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_15 = pd.concat(list,axis=0)
    return data_15

def return_3():
    list = []
    for i in range(0,20):
        Q1 = abnormal_3(20000331+i*10000,20000404+i*10000)
        Q2 = abnormal_3(20000630+i*10000,20000704+i*10000)
        Q3 = abnormal_3(20000930+i*10000, 20001004+i*10000)
        Q4 = abnormal_3(20001231+i*10000, 20000104+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_30 = pd.concat(list,axis=0)
    return data_30

def return_5():
    list = []
    for i in range(0,20):
        Q1 = abnormal_5(20000331+i*10000,20000506+i*10000)
        Q2 = abnormal_5(20000630+i*10000,20000806+i*10000)
        Q3 = abnormal_5(20000930+i*10000, 20001106+i*10000)
        Q4 = abnormal_5(20001231+i*10000, 20000206+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_45 = pd.concat(list,axis=0)
    return data_45

def return_7():
    list = []
    for i in range(0,20):
        Q1 = abnormal_7(20000331+i*10000,20000508+i*10000)
        Q2 = abnormal_7(20000630+i*10000,20000808+i*10000)
        Q3 = abnormal_7(20000930+i*10000, 20001108+i*10000)
        Q4 = abnormal_7(20001231+i*10000, 20000208+(i+1)*10000)
        list.append(Q1)
        list.append(Q2)
        list.append(Q3)
        list.append(Q4)
    data_60 = pd.concat(list,axis=0)
    return data_60


if __name__ == '__main__':
    data_1 = return_1()
    data_3 = return_3()
    data_5 = return_5()
    data_7 = return_7()
    targets = data_1.merge(data_3)
    targets = pd.merge(targets,data_5)
    targets = pd.merge(targets, data_7)
    targets.to_csv('./Stock Return/targets.csv')



