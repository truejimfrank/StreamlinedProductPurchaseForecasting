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

# my functions
from error_split import simple_splitter

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
    drop_percent = 0.2
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
    model.compile(loss="mse", optimizer="rmsprop") # or try 'adam'
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



def predictions_run_plot(X_test, y_test):
    predicted = lstm.predict_point_by_point(model, X_test)
    plot_results(predicted, y_test, 'Sine_wave-predict_one_point_ahead')
    # unsure yet if I need the sequence predict
    predicted_full = lstm.predict_sequence_full(model, X_test, seq_len)
    plot_results(predicted_full, y_test, 'Sine_wave-predict_full_sequence_from_start_seed') 

if __name__ == '__main__':
    # this block used for testing
    rnn_seq_len = 7
    dfday = pd.read_pickle('../../data/time_ecom/dfday.pkl')

    fullarray, thescaler = scale_df2array(dfday)
    # train, test = simple_splitter(fullarray, test_len=28)
    train, test = simple_splitter(fullarray, test_len=28 + 1 + rnn_seq_len)

    # shapes are with rnn_seq_len = 14
    # ((79, 14, 1), 'Xshape  :  yshape', (79, 1))
    X_train, y_train = array2rnnformat(train, seq_len=rnn_seq_len)

    # ((13, 14, 1), 'Xshape  :  yshape', (13, 1)) IF USING testshort
    # ((28, 14, 1), 'Xshape  :  yshape', (28, 1)) IF COMPING FOR seq_len 
    X_test, Y_test = array2rnnformat(test, seq_len=rnn_seq_len, do_shuffle=False)

    layers4lstm = [14, 52, 1]
    themodel = lstm_compile(layer_control=layers4lstm, dropo=False)
    lstm_fit(themodel, X_train, y_train, ep=1, batch=14)
    themodel.summary()
    
