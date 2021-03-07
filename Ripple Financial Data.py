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
start_date = "2020-03-01"
end_date = "2021-03-05"
ripple = data.get_data_yahoo(stock_code, start_date, end_date)
ripple.to_csv(r'C:\Users\Zongyu Lyu\Desktop\Raw Ripple price.csv')

"""
Calculate daily return and volatility for every 30 days windows
"""
ripple = pd.read_csv(r'C:\Users\Zongyu Lyu\Desktop\Raw Ripple Price.csv')
daily_return = ripple['Adj Close'].pct_change(1)
# daily return
values = daily_return.values
# volatility for every 30 days windows
volatility = []
for day in range(1,1000):
    if day+30 > len(values) - 1:
        break
    volatility.append(np.std(values[day:day+30], ddof=1))
volatility = pd.DataFrame({'30 days volatility':volatility})

ripple['Adj Close'].plot(grid=True, figsize=(8,5))
daily_return.plot(grid=True, figsize=(8,5))
volatility.plot(grid=True, figsize=(8,5))
data_cleaning = pd.DataFrame({'Date':ripple['Date'],
                              'Adj Close':ripple['Adj Close'],
                              'daily return':daily_return})
cleaned = pd.concat([data_cleaning,volatility],axis=1)
cleaned.to_csv(r'C:\Users\Zongyu Lyu\Desktop\Cleaned Ripple Financial Data.csv',index=False)









