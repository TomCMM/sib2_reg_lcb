
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model


if __name__=='__main__':
    inpath = "/home/tom/Documentos/data/"
    loading = pd.read_csv(inpath+"loadings_pca_sta_coldpool.csv", index_col=0)
    loading_T = pd.read_csv(inpath+"loadings_pca_sta_coldpool_T.csv", index_col=0)
    scores = pd.read_csv(inpath+"scores_pca_sta_coldpool.csv", index_col=0)
    scores_T = pd.read_csv(inpath+"scores_pca_sta_coldpool_T.csv", index_col=0)
    
    loading_true = pd.read_csv(inpath+"loadings_pca_sta_coldpool_fullperiod.csv", index_col=0)
    print loading    
    print loading_true
#     
#     loading =( loading - loading.mean(axis=0)) / loading.std(axis=0) # another way to standardise 
#     
#     loading =loading + loading.min(axis=0).abs()
# 
#     loading_true = (loading_true - loading_true.mean(axis=0)) / loading_true.std(axis=0) 
#     loading_true=loading_true + loading_true.min(axis=0).abs()
#     
    
    
    X_train = loading.loc[:,['1','2','3','4','5','6']].values[:,:]
    Y_train = loading_true.loc[:,'3'].values[:]
#     X_test = loading.loc[:,['1','2','3','4','5','6']].values[:10,:]
#     Y_test = loading_true.loc[:,'3'].values[:10]
    


    
    lr = linear_model.LinearRegression()
    
    lr.fit(X_train, Y_train)
    
#     Y_predict = lr.predict(X_test)
#     
#     
    plt.scatter(Y_train, lr.predict(X_train))
    plt.show()
    
#     plt.scatter(Y_test,Y_predict)
#     plt.show()
#     
    
    
    
    
    