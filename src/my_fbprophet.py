import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet

def train_prophet(df):
    """train FB prophet model on DF 
    with 'ds' 'y' format
    changepoint_prior_scale default is 0.05
    increasing this parameter gives the trend more flexibility
    returns fitted FB prophet model
    """
    m = Prophet(changepoint_prior_scale=0.05)
    return m.fit(df)

def predict_horizon(m, horizon=28):
    """use fitted fb model to create forecast DF
    forecast is periods past training data
    """
    dffuture = m.make_future_dataframe(periods=horizon)
    dfforecast = m.predict(dffuture)



if __name__ == '__main__':
# open pickled dataframe from data2frame.py
    dfday = pd.read_pickle('../../data/time_ecom/dfday.pkl')