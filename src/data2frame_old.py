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
    return df.copy()

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
    """takes raw DF from events.csv and outputs daily purchases DF
    dfday formatted for FB prophet: 'ds', 'y'  """
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

def format_df_cat_pickle(cat_prod):
    """make df_cat DataFrame for joining to dfevents
    load category DataFrame made in item_categories.py """
    cat_prod['value'] = cat_prod['value'].astype('int32')
    cat_prod.rename(columns={'value':'category'}, inplace=True)
    return cat_prod.copy()

def join_categories(dfevents, dfcat, dftree):
    """ Input: raw dfevents, dfcat from format function, raw dftree
    join category column
    join parent column
    formatting of data into desired types
    """
    df = unix_time_convert(dfevents)
    df = df[df['event'] == 'transaction'].sort_values('timestamp')
    joined = df.join(dfcat.set_index('itemid'), on='itemid')
    joined['category'] = joined['category'].fillna(-1) # copy slice error??
    joined['category'] = joined['category'].astype('int32')
    # 23838 - 22547 = 1291 extra records from items with 2 categories
    joined = joined.drop_duplicates()
    # now join the parent column from dftree
    dftree.rename(columns={'parentid':'parent'}, inplace=True)
    parent = dftree.dropna()  #  from 1669 to 1644
    parent['parent'] = parent['parent'].astype('int32') # copy slice error??
    joined = joined.join(parent.set_index('categoryid'), on='category')
    joined['parent'] = joined['parent'].fillna(-2)
    joined['parent'] = joined['parent'].astype('int32')
    return joined.copy()

def cat2day(dfcat, cat_int, fb=False):
    """takes dfcat 23838 and outputs daily purchases DF
    choose cat_int to only count that category ID
    """
    # filter category
    dfcat = dfcat[dfcat['category'] == cat_int].sort_values('timestamp')
    # rename for working with FB prophet
    dfcat.rename(columns={'event':'y'}, inplace=True)
    dfcat.rename(columns={'timestamp':'ds'}, inplace=True)
    # select columns for easier AGG
    dfcat = dfcat[['ds', 'y', 'category']]
    # resample to daily frequency and count transactions
    dfday = dfcat.resample('D', on="ds").count() 
    # make and join 139 df to pad missing head and tail
    dr = pd.date_range(start='2015-05-02', end='2015-09-17', freq='D')
    dfmake = pd.DataFrame(index=dr)
    dfmake = dfmake.join(dfday)
    # reset category to be correct
    dfmake['category'] = cat_int
    dfmake['y'] = dfmake['y'].fillna(0)
    dfmake['ds'] = dfmake['ds'].fillna(0)
    dfmake = dfmake.astype('int32')  #  the join made values floats
    # make ds, y format for FB
    if fb:  
        dfmake = dfmake[['y']].reset_index().rename(columns={'index': 'ds'})
    # remove partial days ( from 139 rows to 137 )
    dfmake = dfmake.iloc[1:-1]
    return dfmake

# topparent = [561, 955, 105, 500, 1095, 805]

if __name__ == '__main__':

    dfevents = pd.read_csv('../../data/ecommerce/events.csv')
    # dfday = events2day(dfevents)
    # daily_purchases_plot(dfday, png_path='../img/default.png')
    # dfday.to_pickle('../../data/time_ecom/dfday.pkl')

# create and save DataFrame with "category" & "parent" columns
    dfcatraw = pd.read_pickle('../../data/ecommerce/prod_with_cat.pkl', compression='zip')
    dfcatformat = format_df_cat_pickle(dfcatraw)
    dftree = pd.read_csv('../../data/ecommerce/category_tree.csv')
    # 1242 unique category ID's
    dfcatparent = join_categories(dfevents, dfcatformat, dftree)
    print(dfcatparent.head())
    dfcatparent.to_pickle('../../data/time_ecom/dfcatparent.pkl', compression='zip')
