import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# from fbprophet import Prophet

def simple_splitter(df, test_len=28):
    """splits off the tail of the dataframe for timeseries split
    returns 2 dataframes"""
    df_train = df[:-test_len]
    df_test = df[-test_len:]
    return df_train, df_test

def mape(y_true, y_pred):
    # mean absolute percent error
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.

def plot_results(true_data, predicted_data, figtitle):
    ''' use when predicting just one analysis window '''
    fig = plt.figure(figsize=(6, 4), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.title(figtitle)
    plt.tight_layout(pad=1)
    plt.savefig(figtitle + '.png', dpi=100)
    plt.close()
    print('Plot saved.')


