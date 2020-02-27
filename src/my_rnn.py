# run in the TensorFlow docker container

import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def df2rnnformat(df, seq_len=14, ):
    """if dfday then len(df)=137
    X is sequece of arrays of len seq_len
    y is the very next singular future value after array X
    """
    for index in range(len(data) - sequence_length):
        pass

