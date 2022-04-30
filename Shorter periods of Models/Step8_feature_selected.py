from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as LGB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score as EVS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Step6_PCA as PCA

df = pd.read_csv('./datasets/numeric_dataset.csv')
del df['Unnamed: 0']
del df['Unnamed: 0.1']
del df['Unnamed: 0.2']
del df['Stkcd']
del df['Accper']
del df['abnormal_return_1']
del df['abnormal_return_3']
del df['abnormal_return_5']
del df['abnormal_return_7']
columns_list = df.columns.values
feature_name = columns_list.tolist()

def features_target():
    PCA.main(3)
    dataset = pd.read_csv('./datasets/numeric_dataset.csv')
    y_1 = dataset['abnormal_return_1'].tolist()
    y_3 = dataset['abnormal_return_3'].tolist()
    y_5 = dataset['abnormal_return_5'].tolist()
    y_7 = dataset['abnormal_return_7'].tolist()
    del dataset['Unnamed: 0']
    del dataset['Unnamed: 0.1']
    del dataset['Unnamed: 0.2']
    del dataset['Stkcd']
    del dataset['Accper']
    del dataset['abnormal_return_1']
    del dataset['abnormal_return_3']
    del dataset['abnormal_return_5']
    del dataset['abnormal_return_7']
    features = dataset.values
    # print(features)
    dataset_1 = {
        'features':features,
        'y_1':y_1
    }
    dataset_3 = {
        'features':features,
        'y_3':y_3
    }
    dataset_5 = {
        'features':features,
        'y_5':y_5
    }
    dataset_7 = {
        'features':features,
        'y_7':y_7
    }
    json = {
        'dataset_1': dataset_1,
        'dataset_3': dataset_3,
        'dataset_5': dataset_5,
        'dataset_7': dataset_7,
    }
    return json

def ad_r2(y_test,result_prediction,train_df):
    p = train_df.shape[1]
    n = train_df.shape[0]
    return (1-((1-r2_score(y_test,result_prediction))*(n-1))/(n-p-1))

def LR(Dataset,dataset_Num,y_num):
    # Training and Testing
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    prediction = LR.predict(X_test)


def CART(Dataset,dataset_Num,y_num):
    # Training and Testing
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    CART = DecisionTreeRegressor(splitter='best')
    CART.fit(X_train, y_train)
    prediction = CART.predict(X_test)
    importance = CART.feature_importances_
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('./Feature_importance/CART_{}.csv'.format(dataset_Num))

def RandomForest(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    RF = RandomForestRegressor(max_features='auto',criterion='squared_error',bootstrap=True)
    RF.fit(X_train,y_train)
    prediction = RF.predict(X_test)
    importance = RF.feature_importances_
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('./Feature_importance/RF_{}.csv'.format(dataset_Num))

def HistGradientBoosting(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    HGB = HistGradientBoostingRegressor(learning_rate=0.3,max_depth=25,max_iter=300)
    HGB.fit(X_train,y_train)
    prediction = HGB.predict(X_test)

    importance = HGB.feature_importances_
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('./Feature_importance/HGB_{}.csv'.format(dataset_Num))



def XgBoost(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    xgb_model = xgb.XGBRegressor(
                                 learning_rate=0.3,
                                 objective='reg:linear',
                                 n_jobs=-1)
    xgb_model.fit(X_train,y_train)
    prediction = xgb_model.predict(X_test)
    importance = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('./Feature_importance/xbg_{}.csv'.format(dataset_Num))


def lgb(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lgb = LGB.LGBMRegressor(objective='regression',num_leaves=2000,
                            learning_rate=0.3,n_estimators=200
    )
    lgb.fit(X_train, y_train, eval_set=[(X_train, y_train)],eval_metric='logloss')
    prediction = lgb.predict(X_test)
    # # features_name
    booster = lgb.booster_
    importance = booster.feature_importance(importance_type='split')
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('./Feature_importance/lgb_{}.csv'.format(dataset_Num))
    # print("The MSE of prediction in lgb is {}".format(MSE(y_test, prediction)))
    # print("The MAE of prediction in lgb is {}".format(MAE(y_test, prediction)))
    # print("The MAPE of prediction in lgb is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in lgb is {}".format(ad_r2(y_test, prediction, X_test)))
    # print("The EVS of prediction in lgb is {}".format(EVS(y_test, prediction)))


def main():
    dataset = {
        '1':['dataset_1','y_1'],
        '3': ['dataset_3', 'y_3'],
        '5': ['dataset_5', 'y_5'],
        '7': ['dataset_7', 'y_7'],
    }
    list = ['1','3','5','7']
    for i in list:
        print('The {} days abnormal return: '.format(i))
        # LR(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')
        # CART(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')
        # lgb(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')
        XgBoost(features_target(), dataset[i][0], dataset[i][1])
        print('-----------------------')
        # HistGradientBoosting(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')
        # RandomForest(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')


if __name__ == '__main__':
    # features_target()
    main()
