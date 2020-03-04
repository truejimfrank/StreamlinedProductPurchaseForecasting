import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot, plot_weekly
from sklearn.metrics import mean_absolute_error

# my functions
from error_split import simple_splitter, mape
from data2frame import parent2day # (df, select_int, par=True, fb=False)

class PurchaseForecast(object):
    """Class to initialize and run FB prophet model on 
    daily purchases in different product categories
    """
    def __init__(self, parent, horizon, df_daily):
        """ parent is integer name of the product category to select
        horizon is future prediction horizon in days : integer """
        self.parent = parent
        self.horizon = horizon
        self.df_daily = df_daily
    # list of target values to hold in object
    #   parent, horizon, df_daily, model, shf_model, forecast,
    #   mape_error, abs_error, percent_growth, low_growth, high_growth
    #   y_true, y_hat  ( from shf error calc )
    def fit(self):
        # runs all the class functions to store values in object
        self._prophet_fit(prior=0.05)
        self._prophet_predict()
        self._shf()
        self._shf_mape()
        self._growth()

    def output_array(self):
        rr = [self.horizon, self.df_daily, self.model, self.shf_model, self.forecast, self.shf_forecast,
            self.mape_error, self.abs_error, self.percent_growth, self.low_growth, self.high_growth]
        return np.array(rr)
    
    def mape(self, y_true, y_pred):
        # mean absolute percent error
        # watch out for divide by zero errors with this one
        return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100., 1)

    def _prophet_fit(self, prior=0.05):
        self.model = Prophet(changepoint_prior_scale=prior, mcmc_samples=250)
        self.shf_model = Prophet(changepoint_prior_scale=prior, mcmc_samples=250)
        print(f"start fit shf model parent={self.parent}")
        self.shf_model.fit(self.df_daily[:-self.horizon])
        print(f"start fit full model parent={self.parent}")
        self.model.fit(self.df_daily)
        print(f"finished fitting parent={self.parent}")
        
    def _prophet_predict(self):
        self.future = self.model.make_future_dataframe(periods=self.horizon)
        self.forecast = self.model.predict(self.future)
        
    def _shf_mape(self):
        self.y_true = self.df_daily.iloc[-self.horizon:, -1].values
        self.shf_forecast = self.shf_model.predict(self.future[:-self.horizon])
        self.y_hat = self.shf_forecast.iloc[-self.horizon:, -1].values
        self.mape_error = self.mape(self.y_true, self.y_hat)
        
    def _shf(self):
        self.y_true = self.df_daily.iloc[-self.horizon:, -1].values
        self.shf_forecast = self.shf_model.predict(self.future[:-self.horizon])
        self.y_hat = self.shf_forecast.iloc[-self.horizon:, -1].values
        self.abs_error = round(mean_absolute_error(self.y_true, self.y_hat) , 1)
        
    def _growth(self):
        # 'trend' is the column to use in forecast
        #  also 'yhat_lower', 'yhat_upper'
        start = self.forecast['trend'].values[-1 - self.horizon]
        mid = self.forecast['trend'].values[-1]
        low = self.forecast['yhat_lower'].values[-1]
        high = self.forecast['yhat_upper'].values[-1]
        self.percent_growth = round(((mid - start) / start) * 100. , 1)
        self.low_growth = round(((low - start) / start) * 100. , 1)
        self.high_growth = round(((high - start) / start) * 100. , 1)
    # END OF CLASS

def all_days2df(df_daily, thehorizon=28):
    """function to make all days forecast dataframe"""
    dictionary = {}
    dictionary[2222] = df_daily  # 2222 placeholder
    model_dictionary = models_dict(dictionary, horiz=thehorizon)
    print_results(model_dictionary)
    return output_dataframe(model_dictionary)

def select_id_array(df, thresh=250):
    """select parentid integers with sufficient
    purchase count and put them in array 
    default 250 selects 30 parent categories """
    parent_counts = df['parent'].value_counts()
    return parent_counts[parent_counts >= thresh].index.values

def daily_df_dict(df, id_array):
    """ runs the parent2day function to create DF for modeling
    input dfcatparent from pickle
    id_array is array of parentid int numbers
    """
    df_dict = {}
    for parent in id_array:
        df_dict[parent] = parent2day(df, parent, par=True, fb=True)
    return df_dict

def models_dict(dict_of_df, horiz=28):
    """Creates dictionary of fitted PurchaseForecast class objects"""
    return_dict = {}
    for key, data in dict_of_df.items():
        forecast_object = PurchaseForecast(key, horiz, data)
        forecast_object.fit()
        return_dict[key] = forecast_object
    return return_dict

def print_results(models_dict):
    """ Options for printing in PurchaseForecast
      parent, horizon, df_daily, model, shf_model, forecast,
      mape_error, abs_error, percent_growth, low_growth, high_growth
      y_true, y_hat  ( from shf error calc )
    """
    for parent, obj in models_dict.items():
        print(obj.parent, " : parent")
        print(obj.horizon, " horizon : abs error", obj.abs_error)
        print("percent growth from trend : ", obj.percent_growth, " : low, high -> ",
                                            obj.low_growth, obj.high_growth)

def output_dataframe(models_dict):
    """outputs dataframe for comparing and graphing forecastmodels"""
    co = ['horizon', 'df_daily', 'model', 'shf_model', 'forecast', 'shf_forecast',
            'mape_error', 'abs_error', 'percent_growth', 'low_growth', 'high_growth']
    key_list = []
    list_of_rows = []
    for key, obj in models_dict.items():
        key_list.append(key)
        list_of_rows.append(obj.output_array())
    frame = pd.DataFrame(data=list_of_rows, index=key_list, columns=co)
    return frame

""" fb forecast columns
    ['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower',
    'trend_upper', 'additive_terms', 'additive_terms_lower',
    'additive_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper',
    'multiplicative_terms', 'multiplicative_terms_lower',
    'multiplicative_terms_upper', 'yhat']"""

def run_stack(df, horizon=28, the30=True):
    if the30:
        parents = select_id_array(df)
    else:
        parents = np.array([561, 955, 105, 500, 1095, 805])
    daily30 = daily_df_dict(df, parents)
    models30 = models_dict(daily30, horiz=horizon)
    print_results(models30)
    return models30

if __name__ == '__main__':
    # this block used for testing
    # open pickled dataframe from data2frame.py
    df = pd.read_pickle('../../data/time_ecom/dfcatparent.pkl', compression='zip')

    prediction_horizon = 28  #  DAYS
    # choose True here if you want to run full top30 comparison, takes a minute
    select_top30 = True

    # if select_top30:
    # # run functions on top30 (this takes a few minutes)
    #     mod_dict = run_stack(df, horizon=prediction_horizon, the30=True)
    #     dfout = output_dataframe(mod_dict)
    #     dfout.to_pickle('../../data/time_ecom/dfout30.pkl', compression='zip')
    # else:    # run functions on top6
    #     mod_dict = run_stack(df, horizon=prediction_horizon, the30=False)
    #     dfout = output_dataframe(mod_dict)
    #     dfout.to_pickle('../../data/time_ecom/dfout6.pkl', compression='zip')

# output a models forecasts dataframe for the full days
    dfday = pd.read_pickle('../../data/time_ecom/dfday.pkl')
    df_forecast = all_days2df(dfday, thehorizon=28)
    df_forecast.to_pickle('../../data/time_ecom/df_full_forecast.pkl', compression='zip')



