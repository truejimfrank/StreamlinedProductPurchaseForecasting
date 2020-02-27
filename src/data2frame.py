import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
# from fbprophet import Prophet

def unix_time_convert(df):
    """convert from unix time format"""
    times=[]
    for i in df['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i//1000.0))
    df['timestamp']=times
    return df

def daily_purchases_plot(dfday, png_path='../img/default.png'):
    """makes and PNG saves the daily purchases raw data plot"""
    plt.style.use('seaborn-whitegrid')
    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15) 
    dfdayplot = dfday.rename(columns={'y': 'Daily Purchases', 
                                    'ds': 'time'}).set_index('time')
    fig, ax = plt.subplots(figsize=(8,5.5))
    dfdayplot.plot(ax=ax, legend=True, color='skyblue') # or 'tan'
    ax.axhline(163.47, label='Mean Purchases= 163.5', linestyle='--')
    # ax.annotate('163.5 Mean Daily Purchases', xy=(0.6, 0.5),  xycoords='axes fraction',
    #             xytext=(0.53, 0.17), textcoords='axes fraction', fontsize=14, 
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=2),
    #             horizontalalignment='right', verticalalignment='top')
    ax.set_ylim(15, 335)
    plt.legend(fontsize=15, loc='best') # 'upper right'
    plt.tight_layout(pad=1)
    plt.savefig(png_path, dpi=100)

def events2day(dfevents):
    """takes raw DF from events.csv and outputs daily purchases DF"""
    dfevents = unix_time_convert(dfevents)
    col_sel = ['timestamp', 'visitorid', 'itemid', 'event']
    # select rows where event = transaction (22457 results)
    dfpurch = dfevents.loc[dfevents['event'] == 'transaction', col_sel]
    dfpurch = dfpurch.sort_values('timestamp')
    # rename for working with FB prophet
    dfpurch.rename(columns={'timestamp':'ds'}, inplace=True)
    # resample to daily frequency and count transactions
    dfday = dfpurch.resample('D', on="ds").count()
    dfday = dfday[['event']]
    dfday.reset_index(inplace=True)
    dfday.rename(columns={'event':'y'}, inplace=True)
    # remove partial days ( from 139 rows to 137 )
    dfday = dfday.iloc[1:-1]
    return dfday


if __name__ == '__main__':

    dfevents = pd.read_csv('../../data/ecommerce/events.csv')
    dfday = events2day(dfevents)
    # daily_purchases_plot(dfday, png_path='../img/default.png')
    # dfday.to_pickle('../../data/time_ecom/dfday.pkl')
