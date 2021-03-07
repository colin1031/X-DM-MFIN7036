# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:02:52 2021

@author: Zongyu Lyu
"""

import pandas as pd
import numpy as np
from pandas_datareader import data, wb

"""
Get data from YahooFinance
"""
stock_code = 'XRP-USD'
start_date = "2020-01-01"
end_date = "2021-03-05"
ripple = data.get_data_yahoo(stock_code, start_date, end_date)
ripple.to_csv(r'C:\Users\Zongyu Lyu\Desktop\Raw Ripple Price.csv')

"""
Calculate daily return and volatility for every 30 days windows
"""
ripple = pd.read_csv(r'C:\Users\Zongyu Lyu\Desktop\Raw Ripple Price.csv')

# daily return
daily_return = ripple['Adj Close'].pct_change(1)

# Volatility for every 30 days windows
values = daily_return.values
volatility = []
for day in range(31,len(values)):
    if day > len(values) - 1:
        break
    volatility.append(np.std(values[day-30:day], ddof=1))
    
# Fill the empty rows of volatility so that it has the same amount of rows as Raw Ripple Price.csv
match_row = list(range(31))+volatility
volatility = pd.DataFrame({'30 days volatility':match_row})

# Data cleanning
data_cleaning = pd.DataFrame({'Date':ripple['Date'],
                              'Adj Close':ripple['Adj Close'],
                              'daily return':daily_return})
cleaned = pd.concat([data_cleaning,volatility],axis=1)

# Select the row from 2020-03-01 to 2021-03-05
cleaned_data = cleaned[61:428]

# Price,return and volatility plotting
cleaned_data['Adj Close'].plot(grid=True, figsize=(8,5))
cleaned_data['daily return'].plot(grid=True, figsize=(8,5))
cleaned_data['30 days volatility'].plot(grid=True, figsize=(8,5))
cleaned.to_csv(r'C:\Users\Zongyu Lyu\Desktop\Cleaned Ripple Financial Data.csv',index=False)









