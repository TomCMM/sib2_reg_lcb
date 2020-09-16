#===============================================================================
# DESCRIPTION
#    this script show a simple implementation of a deep leaning system 
#    to predict combined PC scores from Global Forecast System predictions 
#===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from toolbox import common_index
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #===========================================================================
    # Get predictors and response
    #===========================================================================
    df_x = pd.read_csv('/home/thomas/phd/framework/predictors/out/gcm_data/local/df_features_scores.csv', index_col=0, parse_dates=True)
    df_y = pd.read_csv('/home/thomas/phd/framework/decomposition/out/regional/scores_ctquv.csv', index_col=0, parse_dates=True)
     
    #===========================================================================
    # Data manipulaation
    #===========================================================================
    df_x.dropna(inplace=True, axis=1)
    df_x = df_x._get_numeric_data()
    df_y = df_y._get_numeric_data()
     
    # get same index
    index = pd.to_datetime(common_index(df_x.index, df_y.index))
    df_x = df_x.loc[index,:]
    df_y = df_y.loc[index,:]
    
    # split training/ test set
    x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values, test_size=0.33, random_state=42)
    
     
#     #===========================================================================
#     # Create Neural Network model
#     #===========================================================================
#     input_dim = len(df_x.columns) # number of input 
#      
#     # simple configurations
#     model = keras.models.Sequential()
#     model.add(keras.layers.Dense(20, input_dim=input_dim, activation='relu'))
#     model.add(keras.layers.Dense(20, activation='relu'))
#     model.add(keras.layers.Dense(1, activation='sigmoid'))
#      
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#  
#     # Fit model
#     model.fit(X_train, y_train[:,0], epochs=150, batch_size=10)
#      
#     #===========================================================================
#     # Evaluate
#     #===========================================================================
#     scores = model.evaluate(X_test, y_test[:,0])
#     print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#      
#     predictions = model.predict(y_test[:,0])
#      
#     print predictions
     
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    
#     # Generate dummy data
#     x_train = np.random.random((1000, 20))
#     y_train = np.random.randint(2, size=(1000, 1))
#     x_test = np.random.random((100, 20))
#     y_test = np.random.randint(2, size=(100, 1))
#
    
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train[:,0],
              epochs=200,
              batch_size=128)
    score = model.evaluate(x_test, y_test[:,0], batch_size=128)
    
    
    
