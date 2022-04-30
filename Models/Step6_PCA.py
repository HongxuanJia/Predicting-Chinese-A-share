import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.2f' % x)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# def Dataset(path):
#     db = pd.read_csv(path)
#     df = db
#     abnormal_return_15 = db['abnormal_return_15']
#     abnormal_return_30 = db['abnormal_return_30']
#     abnormal_return_45 = db['abnormal_return_45']
#     abnormal_return_60 = db['abnormal_return_60']
#     abnormal_return_75 = db['abnormal_return_75']
#     abnormal_return_90 = db['abnormal_return_90']
#     # print(db.head(10))
#     del db['Unnamed: 0']
#     del db['Stkcd']
#     del db['Accper']
#     # del db['Typrep']
#     # del db['Stknme']
#     del db['abnormal_return_15']
#     del db['abnormal_return_30']
#     del db['abnormal_return_45']
#     del db['abnormal_return_60']
#     del db['abnormal_return_75']
#     del db['abnormal_return_90']
#     features = db
#     # print(db.head(100))
#     json = {
#         'features':features,
#         'df': db
#     }
#     return json

def PCAnalysis(path,k):
    db = pd.read_csv(path)
    # print(db.head(10))
    returns = db[['Unnamed: 0','Stkcd','Accper','abnormal_return_15','abnormal_return_30','abnormal_return_45','abnormal_return_60','abnormal_return_75','abnormal_return_90']]
    du = db['Unnamed: 0']
    ds = db['Stkcd']
    da = db['Accper']
    del db['Unnamed: 0']
    del db['Stkcd']
    del db['Accper']
    # del db['Typrep']
    # del db['Stknme']
    del db['abnormal_return_15']
    del db['abnormal_return_30']
    del db['abnormal_return_45']
    del db['abnormal_return_60']
    del db['abnormal_return_75']
    del db['abnormal_return_90']
    features = db
    # print(features.head(),features.shape)
    pca = PCA(n_components=k)
    pca.fit(features)
    features = pca.transform(features)
    scaler = MinMaxScaler()
    scaler.fit(features)
    X_scaler = scaler.transform(features)  # ndarray
    features = pd.DataFrame(X_scaler)
    features['Unnamed: 0'] = du
    features['Stkcd'] = ds
    features['Accper'] = da
    dataset = pd.merge(features,returns)
    del dataset['Unnamed: 0']
    # del dataset['Stkcd']
    # del dataset['Accper']
    # print(dataset.head(10),dataset.shape)
    dataset.set_index(['Stkcd','Accper'])
    dataset.to_csv('./datasets/pca_datasets.csv')
    # db = shuffle(dataset)
    # trian_data = db[:int(len(db) * 0.7)]
    # test_data = db[int(len(db) * 0.7):int(len(db) * 0.8)]
    # validate_data = db[int(len(db) * 0.8):]
    # trian_data.to_csv('./datasets/train_data.csv')
    # test_data.to_csv('./datasets/test_data.csv')
    # validate_data.to_csv('./datasets/validate_data.csv')
    # X_train = db[]

def PCAnalysis_without_macro(path,k):
    db = pd.read_csv(path)
    # print(db.head(10))
    returns = db[['Unnamed: 0','Stkcd','Accper','abnormal_return_15','abnormal_return_30','abnormal_return_45','abnormal_return_60','abnormal_return_75','abnormal_return_90']]
    du = db['Unnamed: 0']
    ds = db['Stkcd']
    da = db['Accper']
    del db['Unnamed: 0']
    del db['Stkcd']
    del db['Accper']
    # del db['Typrep']
    # del db['Stknme']
    del db['abnormal_return_15']
    del db['abnormal_return_30']
    del db['abnormal_return_45']
    del db['abnormal_return_60']
    del db['abnormal_return_75']
    del db['abnormal_return_90']
    features = db
    # print(features.head(),features.shape)
    pca = PCA(n_components=k)
    pca.fit(features)
    features = pca.transform(features)
    scaler = MinMaxScaler()
    scaler.fit(features)
    X_scaler = scaler.transform(features)  # ndarray
    features = pd.DataFrame(X_scaler)
    features['Unnamed: 0'] = du
    features['Stkcd'] = ds
    features['Accper'] = da
    dataset = pd.merge(features,returns)
    del dataset['Unnamed: 0']
    # del dataset['Stkcd']
    # del dataset['Accper']
    # print(dataset.head(10),dataset.shape)
    dataset.set_index(['Stkcd','Accper'])
    dataset.to_csv('./datasets/pca_datasets_without_macro.csv')
    # db = shuffle(dataset)
    # trian_data = db[:int(len(db) * 0.7)]
    # test_data = db[int(len(db) * 0.7):int(len(db) * 0.8)]
    # validate_data = db[int(len(db) * 0.8):]
    # trian_data.to_csv('./datasets/train_data.csv')
    # test_data.to_csv('./datasets/test_data.csv')
    # validate_data.to_csv('./datasets/validate_data.csv')
    # X_train = db[]

def main(k):
    # PCAnalysis('./datasets/numeric_dataset.csv',k)
    PCAnalysis_without_macro('./datasets/numeric_dataset_without_macro.csv',k)

if __name__ == '__main__':
    # FeatureSelection(Dataset('./datasets/train_data.csv'))
    main(8)


# def Dataset(path):
#     db = pd.read_csv(path)
#     abnormal_return_15 = db['abnormal_return_15']
#     abnormal_return_30 = db['abnormal_return_30']
#     abnormal_return_45 = db['abnormal_return_45']
#     abnormal_return_60 = db['abnormal_return_60']
#     abnormal_return_75 = db['abnormal_return_75']
#     abnormal_return_90 = db['abnormal_return_90']
#     # print(db.head(10))
#     del db['Unnamed: 0']
#     del db['Stkcd']
#     del db['Accper']
#     # del db['Typrep']
#     # del db['Stknme']
#     del db['abnormal_return_15']
#     del db['abnormal_return_30']
#     del db['abnormal_return_45']
#     del db['abnormal_return_60']
#     del db['abnormal_return_75']
#     del db['abnormal_return_90']
#     features = db
#     # print(db.head(100))
#     json = {
#         'features':features,
#         'abnormal_return_15': abnormal_return_15,
#         'abnormal_return_30': abnormal_return_30,
#         'abnormal_return_45': abnormal_return_45,
#         'abnormal_return_60': abnormal_return_60,
#         'abnormal_return_75': abnormal_return_75,
#         'abnormal_return_90': abnormal_return_90
#     }
#     return json

# def FeatureSelection(Dataset):
#     train_features = Dataset["features"]
#     df_X = train_features
#     X_train = np.array(train_features)
#     y_train_15 = Dataset["abnormal_return_15"]
#     df_y = y_train_15
#     # the importance of each feature
#     rf = RandomForestRegressor()
#     rf.fit(X_train,y_train_15.ravel())
#     y_train_15 = np.array(y_train_15).reshape(-1,1)
#     importance = rf.feature_importances_
#     # print(importance)
#     plt.bar(df_X.columns,importance)
#     select = SelectFromModel(
#         RandomForestRegressor(n_estimators=50,random_state=150),
#         threshold="median",max_features=150)
#     select.fit(X_train, y_train_15.ravel())
#     mask = select.get_support()
#     columns = df_X.columns[mask].values.tolist()
#     # visualize the mask -- black is True, white is False
#     plt.matshow(mask.reshape(1, -1), cmap='gray_r')
#     plt.xlabel("Sample index")
#     plt.show()
#     # print(X_train_Selected.shape)    #
#     # print(type(X_train_Selected))    # ndarray
#     original_columns = df_X.columns.values.tolist()
#     for i in original_columns:
#         if i not in columns:
#             del df_X[i]
#         else:
#             continue
#     print("The important features:\n {}".format(columns))
#     json = {
#         'features':df_X,
#         'abnormal_return_15': Dataset["abnormal_return_15"],
#         'abnormal_return_30': Dataset["abnormal_return_30"],
#         'abnormal_return_45': Dataset["abnormal_return_45"],
#         'abnormal_return_60': Dataset["abnormal_return_60"],
#         'abnormal_return_75': Dataset["abnormal_return_75"],
#         'abnormal_return_90': Dataset["abnormal_return_90"]
#     }
#     # print("Samples' shape: {} \n Targets' shape: {}".format(df_X.shape,df_y.shape))
#     return json

# def NonPCAnalysis(Dataset):
#         X = Dataset["features"]
#         y_15 = Dataset["abnormal_return_15"]
#         y_30 = Dataset["abnormal_return_30"]
#         y_45 = Dataset["abnormal_return_45"]
#         y_60 = Dataset["abnormal_return_60"]
#         y_75 = Dataset["abnormal_return_75"]
#         y_90 = Dataset["abnormal_return_90"]
#         y_15 = y_15.values.reshape(-1, 1)
#         y_30 = y_30.values.reshape(-1, 1)
#         y_45 = y_45.values.reshape(-1, 1)
#         y_60 = y_60.values.reshape(-1, 1)
#         y_75 = y_75.values.reshape(-1, 1)
#         y_90 = y_90.values.reshape(-1, 1)
#         json = {
#             "X_scaler": X,
#             "y_15": y_15,
#             "y_30": y_30,
#             "y_45": y_45,
#             "y_60": y_60,
#             "y_75": y_75,
#             "y_90": y_90,
#         }
#         print(X.head(100))
#         return json