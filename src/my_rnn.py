# run in the TensorFlow docker container

import pandas as pd 
import numpy as np
import datetime
import time
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
import tensorflow as tf

# my functions
from error_split import simple_splitter, mape, plot_results

np.random.seed(9)

def scale_df2array(df):
    """input dataframe formatted for FBprophet
    columns = ['ds', 'y']
    returns array.shape=(len(df), 1) and the fit scaler for inverse transform
    """
    a_ray = np.array(df['y'].values).reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_array = scaler.fit_transform(a_ray)
    return scaled_array, scaler

def array2rnnformat(scaled_array, seq_len=14, do_shuffle=True):
    """if dfday then len(df)=137
    default len(df_test)=28 so len(df_train)=109
    If seq_len=14, shortens output to len(X_lstm)=94  -(14 + 1)
    X is sequece of arrays of len seq_len
    y is the very next singular future value after array X
    x=len(df), y=seq_len, z=num features (parallel timeseries)
    output = arrayshape(x,y,z)  keras LSTM wants this
    """
    # add 1 to seq_len because 1 is removed later to make y_data
    seq_len += 1
    lstm_array = []  #  list of arrays
    for index in range(scaled_array.shape[0] - seq_len):
        lstm_array.append(scaled_array[index : index + seq_len])
    # convert the list to array
    lstm_array = np.array(lstm_array)
    # shuffle the array of arrays
    if do_shuffle:
        np.random.shuffle(lstm_array)
    # take the last 'column' value, this becomes y
    X_lstm = lstm_array[:, :-1]
    y_lstm = lstm_array[:, -1]
    # reshape X to (x,y,z) for LSTM input
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    return X_lstm, y_lstm

# return_sequences: Boolean. Whether to return the last output 
# in the output sequence, or the full sequence.
# Used to keep (x,y,z) shape for next LSTM layer

def lstm_compile(layer_control=[14,52, 1], dropo=False):
    """layer_control= [seq_len, #hidden_neurons, output_dim]
    2-layer LSTM RNN model
    """
    drop_percent = 0.4
    model = Sequential()

    model.add(LSTM(
        units = layer_control[0],
        return_sequences=True))
    # model.add(Activation("tanh"))  UNNEEDED LSTM has its own activation
    if dropo:
        model.add(Dropout(drop_percent))

    model.add(LSTM(
        units=layer_control[1],
        return_sequences=False))
    # model.add(Activation("tanh"))  UNNEEDED LSTM has its own activation
    if dropo:
        model.add(Dropout(drop_percent))

    model.add(Dense(
        units=layer_control[2]))
    model.add(Activation("linear"))

    start = time.time()
    # loss="mse" to start // mape, poisson(can break solver), logcosh(flattens)
    # mean_absolute_error
    opti = tf.keras.optimizers.RMSprop(learning_rate=0.0002)  # default is 0.001
    model.compile(loss="mse", optimizer=opti) # try 'rmsprop' or 'adam'
    print("Model Compile Time : ", time.time() - start)
    return model

def lstm_fit(model, X_train, y_train, ep=1, batch=14):
    """fit the model to the training data. record processing time."""
    start = time.time()
    model.fit(X_train, y_train, epochs=ep, batch_size=batch)
    print('Model fit duration : ', time.time() - start)

def standard_predict(model, scaler, X_test):
    """input X data must have same seq_len as fitted model"""
    y_hat = model.predict(X_test)
    y_hat = scaler.inverse_transform(y_hat)
    return y_hat

if __name__ == '__main__':
    # this block used for testing
    rnn_seq_len = 8
    dfday = pd.read_pickle('../../data/time_ecom/dfday.pkl')

    fullarray, thescaler = scale_df2array(dfday)
    # train, test = simple_splitter(fullarray, test_len=28)
    train, test = simple_splitter(fullarray, test_len=28 + 1 + rnn_seq_len)

    # shapes are with rnn_seq_len = 14
    # ((79, 14, 1), 'Xshape  :  yshape', (79, 1))
    X_train, y_train = array2rnnformat(train, seq_len=rnn_seq_len)

    # ((13, 14, 1), 'Xshape  :  yshape', (13, 1)) IF USING testshort
    # ((28, 14, 1), 'Xshape  :  yshape', (28, 1)) IF COMPING FOR seq_len 
    X_test, y_test = array2rnnformat(test, seq_len=rnn_seq_len, do_shuffle=False)
    y_test = thescaler.inverse_transform(y_test)

    # layers4lstm = [rnn_seq_len, 50, 1]
    layers4lstm = [200, 200, 1]
    themodel = lstm_compile(layer_control=layers4lstm, dropo=True)
    lstm_fit(themodel, X_train, y_train, ep=9, batch=99)
    themodel.summary()

    y_hat = standard_predict(themodel, thescaler, X_test)
    print(y_hat.shape[0], " : len of prediction array")
    # for row in y_hat:
    #     print(row[0])
    error = round(mape(y_test, y_hat), 1)
    print("mape RNN : ", error)

    # rnntitle = 'RNN forecast with sequence={0} error={1}%'.format(str(rnn_seq_len), error)
    rnntitle = 'RNN forecast with {0}%'.format(error) + ' error'
    plot_results(y_test, y_hat, rnntitle)
