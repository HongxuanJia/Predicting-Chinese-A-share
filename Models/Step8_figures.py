import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format',lambda x: '%.4f' % x)

if __name__ == '__main__':
    df = pd.read_excel('./figures/feature_selected.xlsx')
    # print(df.head(10))
    del df['featureName']
    df.set_index(['Description'])
    print(df.head(10))
    # del df['Description']
    sns.heatmap(df,vmax=0.12,vmin=0.001,cmap=sns.cubehelix_palette(as_cmap=True))
    plt.show()
    # plt.matshow(df['CART','RF','XgBoost','LightGB'])
    # plt.show()



