import pandas as pd
import numpy as np

"""
load and merge product category files
outputs compressed pickle dataframes
1) product ID with associated category
2) unique product ID's from these category files
note: There are significantly less unique products in the events.csv
"""

items1=pd.read_csv('../data/ecommerce/item_properties_part1.csv')
items2=pd.read_csv('../data/ecommerce/item_properties_part2.csv')
print(items2.info())
items=pd.concat([items2,items1])

arrid = items['itemid'].unique()
uni_prod_id = pd.DataFrame(arrid)
# mask1 = items['itemid'] == arrid
mask2 = items['property'] == 'categoryid'
# combo = 
arr_data = items.loc[mask2, ['itemid', 'value']]
# wrangle = pd.DataFrame(data= , columns=)
# wrangle.to_pickle('../data/ecommerce/products.pkl', compression='zip')
arr_data.to_pickle('../data/ecommerce/prod_with_cat.pkl', compression='zip')
uni_prod_id.to_pickle('../data/ecommerce/prod_unique.pkl', compression='zip')

