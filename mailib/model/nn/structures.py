"""
This module contains different Neural Network structures
in Keras as examples

"""

from keras.models import Sequential, load_model
from keras.layers import GRU,Dense,Flatten,Dropout, AlphaDropout,Input
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.models import load_model, Model
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt


def CCN_RNN():
    model = Sequential()

    model.add(Conv2D(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), filters=50, kernel_size=(3, 3),
                     strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Reshape(target_shape=(16 * 16, 50)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1)

def CCN_RNN_V2():
    input_layer = Input(shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    conv_layer = Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same')(input_layer)
    activation_layer = Activation('relu')(conv_layer)
    pooling_layer = MaxPooling2D(pool_size = (2,2), padding = 'same')(activation_layer)
    flatten = Flatten()(pooling_layer)
    dense_layer_1 = Dense(100)(flatten)

    reshape = Reshape(target_shape = (X_train.shape[1]*X_train.shape[2], X_train.shape[3]))(input_layer)
    lstm_layer = LSTM(50, return_sequences = False)(reshape)
    dense_layer_2 = Dense(100)(lstm_layer)
    merged_layer = concatenate([dense_layer_1, dense_layer_2])
    output_layer = Dense(10, activation = 'softmax')(merged_layer)

    model = Model(inputs = input_layer, outputs = output_layer)

    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

def stacked_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(49, 1), return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(49, 1), return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def stacked_simple_rnn_local():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(49, 1),
                        return_sequences=True))  # return_sequences parameter has to be set True to stack
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def deep_lstm_local():
    model = Sequential()
    model.add(LSTM(20, input_shape=(49, 1), return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def vanilla_rnn():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def stacked_vanilla_rnn():
    model = Sequential()
    model.add(SimpleRNN(2, input_shape=(49, 1),
                        return_sequences=True))  # return_sequences parameter has to be set True to stack
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def RNN_ribeirao(x_train, y_train, epochs, model_path):


    model = Sequential()
    # model.add(Dense(12,activation='relu'))
    model.add(GRU(20,input_shape=x_train.shape[1:], return_sequences=True, bias_initializer='zeros', use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(50,activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',metrics=['accuracy'], optimizer='adam')
    model.summary()



    callbacks = [ModelCheckpoint(filepath=model_path, monitor="val_mean_squared_error", save_best_only=True)]  # TODO save best model base on validation test
    # model = Model(inputs=[main_input], outputs=out)

    model.compile(loss='mean_squared_error',metrics=[metrics.mae, metrics.mse], optimizer='adam')
    # validation_split=0.2
    history = model.fit(x_train, y_train, epochs=epochs, shuffle=False, verbose=2, batch_size=12, callbacks=callbacks)

    # history = model.fit(x_train, y_train,  epochs=epochs,validation_split = validation_split, shuffle=True, verbose=2, batch_size=12, callbacks=callbacks)



    return model, history


def DNN_ribeirao(x_train, y_train, epochs, model_path, plot=False):

    """
    Deep Artificial Neural Network developed to predict the PC scores
    over the Ribeirao Das Posses

    :param x_train:
    :param y_train:
    :return: fitted model
    """

    # ===========================================================================
    # Neural Network structure
    # ===========================================================================
    # model = Sequential()
    # model.add(Dense(units=50, activation='selu', input_dim=x_train.shape[1], init='lecun_uniform')) # use_bias=True,bias_initializer='zeros'))
    # # model.add(Dropout(0.2))
    # model.add(Dense(units=50, activation='sigmoid', init='lecun_uniform'))
    # model.add(Dropout(0.4))
    # model.add(Dense(units=1, activation='sigmoid', init='lecun_uniform'))
    # 
    # callbacks = [ModelCheckpoint(filepath=model_path, monitor="val_mean_squared_error", save_best_only=True)]  # TODO save best model base on validation test
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.mae, metrics.mse])
    # # fit model with training set
    # history = model.fit(x_train, y_train , epochs=epochs, batch_size=12, shuffle=True, verbose=2, callbacks=callbacks)
    # 
    # 


    main_input = Input(shape=x_train.shape[1:], name='main_input')
    # gru = Dropout(0.2)(gru)
    # gru = Flatten()(gru)
    #
    # aux_input = Input(shape=(7,), init='lecun_uniform', name='aux_input')
    # # aux = Dense(5, activation='relu')(aux_input)
    # merged = concatenate([gru, aux_input])

    # out = AlphaDropout(0.2)(merged)
    # out = BatchNormalization()(main_input)
    out = Dense(100,activation='relu', bias_initializer='zero', use_bias=True)(main_input)# ,use_bias=True,bias_initializer='zeros')(main_input)#  use_bias=True,bias_initializer='zeros')(main_input)
    # out = AlphaDropout(0.2)(out)
    # out = Dropout(0.4)(out)
    # out = Dense(100,activation='relu')(out)
    # out = Dropout(0.3)(out)
    # out = Dense(50,activation='sigmoid')(out)
    # out = Dense(50,activation='relu')(out)
    # out = Dropout(0.2)(   out)
    out = Dense(50,activation='sigmoid')(out)
    out = Dropout(0.2)(out)
    # out = AlphaDropout(0.2)(out)
    #out = Dense(80,activation='relu')(out)
    # out = BatchNormalization()(out)
    out = Dense(1, activation='sigmoid')(out)
    # out = LeakyReLU(alpha=0.3)(out)

    callbacks = [ModelCheckpoint(filepath=model_path, monitor="val_mean_squared_error", save_best_only=True)]  # TODO save best model base on validation test
    model = Model(inputs=[main_input], outputs=out)

    model.compile(loss='mean_squared_error',metrics=[metrics.mae, metrics.mse], optimizer='adam')
    validation_split=0.20
    history = model.fit(x_train, y_train,  epochs=epochs,validation_split = validation_split, shuffle=True, verbose=2, batch_size=12, callbacks=callbacks)

    split_at = int(y_train.shape[0] * (1 - validation_split))
    # y_obs = pd.DataFrame(y, index=idx_y)

    print(r2_score(y_train, model.predict(x_train)))
    print(r2_score(y_train[split_at:], model.predict(x_train[split_at:])))

    print(mean_squared_error(y_train, model.predict(x_train)))
    print(mean_squared_error(y_train[split_at:], model.predict(x_train[split_at:])))

    if plot:
        
        print('Entering plot function' + '*' * 10)
        plt.figure(figsize=(8, 6))
        plt.plot(model.predict(x_train), alpha=0.5, c='r', label='predicted')
        plt.plot(y_train, alpha=0.5, c='b', label='observed')
        plt.legend()


    return model, history

def DNN_ribeirao_irr(x_train, y_train, epochs, model_path, plot=False):

    """
    Deep Artificial Neural Network developed to predict the PC scores
    over the Ribeirao Das Posses

    :param x_train:
    :param y_train:
    :return: fitted model
    """

    # ===========================================================================
    # Neural Network structure
    # ===========================================================================
    # model = Sequential()
    # model.add(Dense(units=50, activation='selu', input_dim=x_train.shape[1], init='lecun_uniform')) # use_bias=True,bias_initializer='zeros'))
    # # model.add(Dropout(0.2))
    # model.add(Dense(units=50, activation='sigmoid', init='lecun_uniform'))
    # model.add(Dropout(0.4))
    # model.add(Dense(units=1, activation='sigmoid', init='lecun_uniform'))
    #
    # callbacks = [ModelCheckpoint(filepath=model_path, monitor="val_mean_squared_error", save_best_only=True)]  # TODO save best model base on validation test
    # model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.mae, metrics.mse])
    # # fit model with training set
    # history = model.fit(x_train, y_train , epochs=epochs, batch_size=12, shuffle=True, verbose=2, callbacks=callbacks)
    #
    #


    main_input = Input(shape=x_train.shape[1:], name='main_input')
    # gru = Dropout(0.2)(gru)
    # gru = Flatten()(gru)
    #
    # aux_input = Input(shape=(7,), init='lecun_uniform', name='aux_input')
    # # aux = Dense(5, activation='relu')(aux_input)
    # merged = concatenate([gru, aux_input])

    # out = AlphaDropout(0.2)(merged)
    # out = BatchNormalization()(main_input)
    out = Dense(100,activation='relu', bias_initializer='zero', use_bias=True)(main_input)# ,use_bias=True,bias_initializer='zeros')(main_input)#  use_bias=True,bias_initializer='zeros')(main_input)

    out = Dense(50,activation='sigmoid')(out)
    out = Dropout(0.2)(out)
    # out = AlphaDropout(0.2)(out)
    #out = Dense(80,activation='relu')(out)
    # out = BatchNormalization()(out)
    out = Dense(1, activation='sigmoid')(out)
    # out = LeakyReLU(alpha=0.3)(out)

    callbacks = [ModelCheckpoint(filepath=model_path, monitor="val_absolute_squared_error", save_best_only=True)]  # TODO save best model base on validation test
    model = Model(inputs=[main_input], outputs=out)

    model.compile(loss='mean_absolute_error',metrics=[metrics.mae, metrics.mse], optimizer='adam')
    validation_split=0.20
    history = model.fit(x_train, y_train,  epochs=epochs,validation_split = validation_split, shuffle=True, verbose=2, batch_size=12, callbacks=callbacks)

    split_at = int(y_train.shape[0] * (1 - validation_split))
    # y_obs = pd.DataFrame(y, index=idx_y)

    print(r2_score(y_train, model.predict(x_train)))
    print(r2_score(y_train[split_at:], model.predict(x_train[split_at:])))

    print(mean_squared_error(y_train, model.predict(x_train)))
    print(mean_squared_error(y_train[split_at:], model.predict(x_train[split_at:])))

    if plot:

        print('Entering plot function' + '*' * 10)
        plt.figure(figsize=(8, 6))
        plt.plot(model.predict(x_train), alpha=0.5, c='r', label='predicted')
        plt.plot(y_train, alpha=0.5, c='b', label='observed')
        plt.legend()


    return model, history

