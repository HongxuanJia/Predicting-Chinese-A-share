from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
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
import Step6_PCA as PCA

def features_target():
    PCA.main(3)
    dataset = pd.read_csv('./datasets/pca_datasets.csv')
    y_1 = dataset['abnormal_return_1'].tolist()
    y_3 = dataset['abnormal_return_3'].tolist()
    y_5 = dataset['abnormal_return_5'].tolist()
    y_7 = dataset['abnormal_return_7'].tolist()
    del dataset['Unnamed: 0']
    # del dataset['Unnamed: 0.1']
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
    print("The MSE of prediction in LR is {}".format(MSE(y_test, prediction)))
    print("The MAE of prediction in LR is {}".format(MAE(y_test, prediction)))
    print("The MAPE of prediction in LR is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in LR is {}".format(ad_r2(y_test, prediction,X_test)))
    print("The EVS of prediction in LR is {}".format(EVS(y_test, prediction)))

def CART(Dataset,dataset_Num,y_num):
    # Training and Testing
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    CART = DecisionTreeRegressor(splitter='best')
    CART.fit(X_train, y_train)
    prediction = CART.predict(X_test)
    print("The MSE of prediction in CART is {}".format(MSE(y_test, prediction)))
    print("The MAE of prediction in CART is {}".format(MAE(y_test, prediction)))
    print("The MAPE of prediction in CART is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in CART is {}".format(ad_r2(y_test, prediction,X_test)))
    print("The EVS of prediction in CART is {}".format(EVS(y_test, prediction)))


def AdaBoost(Dataset,String):
    trainData = Dataset['data_train']
    Features = trainData["X_scaler"]
    Targets = trainData[String]
    train_feature = Features
    train_target = Targets
    Adaboost = AdaBoostRegressor(DecisionTreeRegressor(),n_estimators=500, learning_rate=0.1,loss='exponential')
    Adaboost.fit(train_feature,train_target.ravel())
    predictData = Dataset['data_validate']
    predict_feature = predictData['X_scaler']
    predict_target = predictData['y_15']
    # the target from label
    predict_target = np.array(predict_target).reshape(-1)
    # print(predict_target)
    prediction = Adaboost.predict(predict_feature)
    # print(prediction)
    print("For {} abnormal return: ".format(String))
    print("The MSE of prediction in AdaBoost is {}".format(MSE(predict_target, prediction)))
    print("The MAE of prediction in AdaBoost is {}".format(MAE(predict_target, prediction)))
    print("The MAPE of prediction in AdaBoost is {}".format(MAPE(predict_target, prediction)))
    print("The r2_score of prediction AdaBoost is {}".format(ad_r2(predict_target, prediction,train_feature)))

def RandomForest(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    RF = RandomForestRegressor(max_features='auto',criterion='squared_error',bootstrap=True)
    RF.fit(X_train,y_train)
    prediction = RF.predict(X_test)
    print("The MSE of prediction in RF is {}".format(MSE(y_test, prediction)))
    print("The MAE of prediction in RF is {}".format(MAE(y_test, prediction)))
    print("The MAPE of prediction in RF is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in RF is {}".format(ad_r2(y_test, prediction, X_test)))
    print("The EVS of prediction in RF is {}".format(EVS(y_test, prediction)))

def HistGradientBoosting(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    HistGradientBoost = HistGradientBoostingRegressor(learning_rate=0.3,max_depth=25,max_iter=300)
    HistGradientBoost.fit(X_train,y_train)
    prediction = HistGradientBoost.predict(X_test)
    print("The MSE of prediction in hgb is {}".format(MSE(y_test, prediction)))
    print("The MAE of prediction in hgb is {}".format(MAE(y_test, prediction)))
    print("The MAPE of prediction in hgb is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in hgb is {}".format(ad_r2(y_test, prediction, X_test)))
    print("The EVS of prediction in hgb is {}".format(EVS(y_test, prediction)))


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
    print("The MSE of prediction in xgb is {}".format(MSE(y_test, prediction)))
    print("The MAE of prediction in xgb is {}".format(MAE(y_test, prediction)))
    print("The MAPE of prediction in xgb is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in xgb is {}".format(ad_r2(y_test, prediction, X_test)))
    print("The EVS of prediction in xgb is {}".format(EVS(y_test, prediction)))


def lgb(Dataset,dataset_Num,y_num):
    dataset = Dataset[dataset_Num]
    X = dataset['features']
    y = dataset[y_num]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    gbm = LGB.LGBMRegressor(objective='regression',num_leaves=2000,
                            learning_rate=0.3,n_estimators=200
    )
    gbm.fit(X_train, y_train, eval_set=[(X_train, y_train)],eval_metric='logloss')
    prediction = gbm.predict(X_test)
    print("The MSE of prediction in lgb is {}".format(MSE(y_test, prediction)))
    print("The MAE of prediction in lgb is {}".format(MAE(y_test, prediction)))
    print("The MAPE of prediction in lgb is {}".format(MAPE(y_test, prediction)))
    print("The r2_score of prediction in lgb is {}".format(ad_r2(y_test, prediction, X_test)))
    print("The EVS of prediction in lgb is {}".format(EVS(y_test, prediction)))

# def main(String):
#     LR(Dataset(),String)
#     print('-------------------------')
#     # SVM(Dataset(),String_list[0])
#     CART(Dataset(),String)
#     print('-------------------------')
#     # AdaBoost(Dataset(),String)
#     # print('-------------------------')
#     RandomForest(Dataset(),String)
#     print('-------------------------')
#     HistGradientBoosting(Dataset(),String)
#     print('-------------------------')

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
        lgb(features_target(), dataset[i][0], dataset[i][1])
        print('-----------------------')
        # XgBoost(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')
        # HistGradientBoosting(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')
        # RandomForest(features_target(), dataset[i][0], dataset[i][1])
        # print('-----------------------')


if __name__ == '__main__':
    # features_target()
    main()
    # LR(features_target(),'dataset_15','y_15')
    # print('-----------------------')
    # CART(features_target(), 'dataset_15', 'y_15')
    # print('-----------------------')
    # lgb(features_target(), 'dataset_15', 'y_15')
    # print('-----------------------')
    # XgBoost(features_target(), 'dataset_15', 'y_15')
    # print('-----------------------')
    # HistGradientBoosting(features_target(), 'dataset_15', 'y_15')
    # print('-----------------------')
