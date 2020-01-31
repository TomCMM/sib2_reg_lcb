"""
DESCRIPTION
    Find the loading weight with neural network
"""




from keras.models import Sequential
from keras.layers import Dense
import numpy

import pandas as pd
import matplotlib.pyplot as plt



if __name__=='__main__':
    inpath = "/home/tom/Documentos/data/"
    loading = pd.read_csv(inpath+"loadings_pca_sta_coldpool.csv", index_col=0)
    loading_T = pd.read_csv(inpath+"loadings_pca_sta_coldpool_T.csv", index_col=0)
    scores = pd.read_csv(inpath+"scores_pca_sta_coldpool.csv", index_col=0)
    scores_T = pd.read_csv(inpath+"scores_pca_sta_coldpool_T.csv", index_col=0)
    
    loading_true = pd.read_csv(inpath+"loadings_pca_sta_coldpool_fullperiod.csv", index_col=0)
    print loading
#     scores_true = pd.read_csv(inpath+"scores_pca_sta_coldpool_fullperiod.csv", index_col=0)
          
#      
#     print loading
#     print loading_true
#     
#     for col in loading.columns:
#         plt.figure()
#         plt.scatter(loading.loc[:,col], loading_true.loc[:,col])
#         plt.show()

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    #===========================================================================
    # TEST neural network regression with Keras
    #===========================================================================
    import numpy
    import pandas
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline


    # load dataset
#     dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
#     dataset = dataframe.values
    # split into input (X) and output (Y) variables
#     X = dataset[:,0:13]
#     Y = dataset[:,13]
    

    loading =( loading - loading.mean(axis=0)) / loading.std(axis=0) # another way to standardise 
    
    loading =loading + loading.min(axis=0).abs()
    print loading.mean(axis=0)

    loading_true = (loading_true - loading_true.mean(axis=0)) / loading_true.std(axis=0) 
    loading_true=loading_true + loading_true.min(axis=0).abs()
    
    
    X_train = loading.loc[:,['1','2','3','4','5','6']].values[:-10,:]
    Y_train = loading_true.loc[:,'3'].values[:-10]
    X_test = loading.loc[:,['1','2','3','4','5','6']].values[-10:,:]
    Y_test = loading_true.loc[:,'3'].values[-10:]
    
    # define base mode
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(4, input_dim=6, init='normal', activation='linear'))
        model.add(Dense(6, init='normal', activation='linear'))
#         model.add(Dense(6, init='normal', activation='linear'))
#         model.add(Dense(4, init='normal', activation='linear'))
#         model.add(Dense(3, input_dim=6, init='normal', activation='tanh'))
#         model.add(Dense(10 init='normal', activation='relu'))
#         model.add(Dense(10, init='normal', activation='linear'))
#         model.add(Dense(2, init='normal', activation='linear'))
        model.add(Dense(1, init='normal'))
        # Compile model
        model.compile(loss="mean_absolute_error", optimizer='adam')
        return model
# 
# 
#     # fix random seed for reproducibility
#     seed = 7
#     numpy.random.seed(seed)
#     # evaluate model with standardized dataset
#     estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
#     
#     print "Estimator"
#     print estimator
#     kfold = KFold(n_splits=10, random_state=seed)
#     results = cross_val_score(estimator, X, Y, cv=kfold)
#     print("Results: %f (%f) MSE" % (results.mean(), results.std()))

    model = baseline_model()
    model.fit(X_train,Y_train, nb_epoch=5000, batch_size=10)
    Y_predict = model.predict(X_test, batch_size=10)
    
    
    
    plt.scatter(Y_train, model.predict(X_train))
    plt.show()
    
    plt.scatter(Y_test,Y_predict)
    plt.show()
    

