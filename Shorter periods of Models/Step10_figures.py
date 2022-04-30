import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def scatter():
    df = pd.read_csv('./datasets/pca_datasets.csv')
    X = df['0'].values.tolist()
    Y = df['1'].values.tolist()
    Z = df['2'].values.tolist()
    plt.figure(figsize=(8,6))  # 设置画布大小
    ax = plt.axes(projection='3d')  # 设置三维轴
    ax.scatter3D(X, Y, Z,c='mediumpurple')  # 三个数组对应三个维度（三个数组中的数一一对应）
    plt.xticks(range(11))  # 设置 x 轴坐标
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 20})
    plt.xlabel('Principal_Component_1')
    plt.ylabel('Principal_Component_2', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Principal_Component_3')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
    plt.savefig('./figures/PCA.png', bbox_inches='tight', dpi=2400)  # 保存图片，如果不设置 bbox_inches='tight'，保存的图片有可能显示不全
    plt.show()

def concat(num):
    CART = pd.read_excel('./return/CART_dataset_{}.xlsx'.format(num))
    HGB = pd.read_excel('./return/HGB_dataset_{}.xlsx'.format(num))
    LR = pd.read_excel('./return/LR_dataset_{}.xlsx'.format(num))
    RF = pd.read_excel('./return/RF_dataset_{}.xlsx'.format(num))
    XGB = pd.read_excel('./return/XGB_dataset_{}.xlsx'.format(num))
    lgb = pd.read_excel('./return/lgb_dataset_{}.xlsx'.format(num))
    dataset1 = pd.concat([CART,HGB,LR,RF,XGB,lgb],axis=1)
    del dataset1['Unnamed: 0']
    dataset1.to_csv('./boxplot/return_{}.csv'.format(num))
    # dataset1.rename(columns={'0':'return'})
    # print(dataset1.head(10))

def boxplot(num):
    abreturn = pd.read_csv('./boxplot/return_{}.csv'.format(num))
    del abreturn['Unnamed: 0']
    abreturn.boxplot()
    # plt.ylabel('abnormal return')
    plt.title('The {}-days'.format(num))
    # plt.show()

if __name__ == '__main__':
    plt.figure(figsize=[8,6])
    plt.subplot(2, 2,1)
    boxplot(1)
    plt.subplot(2, 2,2)
    boxplot(3)
    plt.subplot(2, 2,3)
    boxplot(5)
    plt.subplot(2, 2,4)
    boxplot(7)
    plt.subplots_adjust(hspace=0.4,wspace=0.3)
    plt.suptitle('The average abnormal return selected from top 80% predicting results during different period')
    plt.savefig('./figures/box.jpg')
    plt.show()