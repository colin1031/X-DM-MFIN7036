# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:02:52 2021

@author: Zongyu Lyu
"""

import pandas as pd
import numpy as np
from pandas_datareader import data, wb
stock_code = 'XRP-USD'
start_date = "2020-03-01"
end_date = "2021-03-01"
ripple = data.get_data_yahoo(stock_code, start_date, end_date)
ripple.to_csv(r'C:\Users\Zongyu Lyu\Desktop\yahoo_data.csv', index=False)

daily_return = ripple['Adj Close'].pct_change(1)
print(daily_return)
ripple['Adj Close'].plot(grid=True, figsize=(8,5))
daily_return.plot(grid=True, figsize=(8,5))