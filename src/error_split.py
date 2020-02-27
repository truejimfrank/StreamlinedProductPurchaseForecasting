import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime
# from fbprophet import Prophet

def simple_splitter(df, test_len=28):
    """splits off the tail of the dataframe for timeseries split
    returns 2 dataframes"""
    df_train = df[:-test_len]
    df_test = df[-test_len:]
    return df_train, df_test

