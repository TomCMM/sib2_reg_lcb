# ===============================================================================
# DESCRIPTION
#    this script show a simple implementation of a deep leaning system
#    to predict combined PC scores from Global Forecast System predictions
# ===============================================================================

# Library
import datetime # Date
import os
import random as rm

import pandas as pd # Data analysis
import numpy as np # Matrix manipulation
import matplotlib # visualitation
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from toolbox import common_index, apply_func # useful toolbox

import tensorflow as tf # Keras backend
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K




if __name__ == '__main__':


    path_main = "/home/thomas/Documents/posdoc/framework"
    path_predictors = '/predictors/out/gcm_data/local/basic_var/df_features_scores.csv'
    path_response = '/fill/out/Ta.csv'
    path_model_save = '/model/out/nn/'

    # Fix specific seed for reproductibility
    os.environ['PYTHONHASHSEED']='0'
    np.random.seed(0)
    rm.seed(0)

    #force tensorflow to use a single thread
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)
    K.set_session(sess)

    # ===========================================================================
    # Get predictors
    # ===========================================================================
    df_x = pd.read_csv(path_main + path_predictors, index_col=0, parse_dates=True)

    # ===========================================================================
    # Get response
    # ===========================================================================

    stadata = pd.read_csv(path_main + path_response, index_col=0, parse_dates=True)
    stadata = stadata.filter(regex='C', axis=1)[:-2]
    stadata.dropna(inplace=True, axis=0, how='all')
    stadata.dropna(inplace=True, axis=1, how='all')
    pca = PCA(4)
    scores = pca.fit_transform(stadata)

    df_y = pd.DataFrame(scores[:, 0], index= stadata.index)


    # df_y = pd.read_csv(path_main + '/decomposition/out/regional/scores_ctquv.csv', index_col=0, parse_dates=True)

    # ===========================================================================
    # Data cleaning/ preprocessing
    # ===========================================================================
    # Drop missing data
    df_x.dropna(inplace=True, axis=0, how='all')
    df_y.dropna(inplace=True, axis=0, how='all')
    df_x.dropna(inplace=True, axis=1, how='all')
    df_y.dropna(inplace=True, axis=1, how='all')
    df_x.dropna(inplace=True, axis=0, how='any')
    df_y.dropna(inplace=True, axis=0, how='any')

    df_x = df_x._get_numeric_data()
    df_y = df_y._get_numeric_data()

    # Get same index
    index = pd.to_datetime(common_index(df_x.index, df_y.index))
    df_x = df_x.loc[index, :]
    df_y = df_y.loc[index, :]

    # Scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    df_x_scaled = scaler.fit_transform(df_x.values)
    df_y_scaled = scaler.fit_transform((df_y).values)

    # Spliting sets
    x_train, x_test, y_train, y_test = train_test_split(df_x_scaled, df_y_scaled, test_size=0.10, random_state=1)

    # ===========================================================================
    # Neural Network structure
    # ===========================================================================
    model = Sequential()
    model.add(Dense(units=24, activation='relu', input_dim=64, use_bias=True,bias_initializer='zeros'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])

    # Visualize weights
    model.get_weights()

    # fit model with training set
    model.fit(x_train, y_train ,epochs=100, batch_size=50, shuffle=True, verbose=2)



    # Evaluate model
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print(loss_and_metrics)

    # Use model to make prediction with the test set
    y_predicted = model.predict(x_test, batch_size=128)

    # # plot
    # plt.plot(y_predicted, c='r')
    # plt.plot(y_test, c='b')
    # plt.show()
    # print('done')

    # model summary
    model.summary()

    #model save
    now = datetime.datetime.now() # get current time
    model.save(path_main + path_model_save + 'nn_model_' + now.strftime("%Y-%m-%d_%H-%M") + '.h5')


