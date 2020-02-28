import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet

# my functions
from error_split import simple_splitter, mape


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
    # dffuture has 'ds' column only
    dffuture = m.make_future_dataframe(periods=horizon)
    dfforecast = m.predict(dffuture)
    """ fb forecast columns
        ['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower',
       'trend_upper', 'additive_terms', 'additive_terms_lower',
       'additive_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper',
       'multiplicative_terms', 'multiplicative_terms_lower',
       'multiplicative_terms_upper', 'yhat']"""
    y_hat = dfforecast.iloc[-horizon: , -1].values
    return y_hat


if __name__ == '__main__':
    # this block used for testing
    # open pickled dataframe from data2frame.py
    dfday = pd.read_pickle('../../data/time_ecom/dfday.pkl')
    thehorizon = 28
    fb_train, fb_test = simple_splitter(dfday, test_len=thehorizon)

    fbmod = train_prophet(fb_train)
    y_hat = predict_horizon(fbmod, horizon=thehorizon)

    y_test = fb_test['y'].values

    print("mape FBprophet : ", mape(y_test, y_hat))
