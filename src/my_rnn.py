# run in the TensorFlow docker container

import pandas as pd 
import numpy as np
import datetime
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


if __name__ == '__main__':
    # this block used for testing
    rnn_seq_len = 14
    dfday = pd.read_pickle('../../data/time_ecom/dfday.pkl')

    fullarray, thescaler = scale_df2array(dfday)
    # ((109, 1), 'train   :   test', (28, 1))
    train, test = simple_splitter(fullarray)

    # ((94, 14, 1), 'Xshape  :  yshape', (94, 1))
    X_train, y_train = array2rnnformat(train)

    # ((13, 14, 1), 'Xshape  :  yshape', (13, 1))
    X_test, Y_test = array2rnnformat(test)


    # model.summary()
