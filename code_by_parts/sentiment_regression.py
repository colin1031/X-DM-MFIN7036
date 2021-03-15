#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import all the packages and set up directory
we suggest put xdm_init.py file in your working directory
"""

from xdm_init import *

#input your working directory
set_dic('')

"""
Read merged data (financial data and sentiment data)
"""
# read 
all_data = pd.read_csv('./final_data.csv')

all_data.columns
all_data['textblob_score_lag_1d'] = all_data['polarty_score_with_textblob'].shift(1)
all_data['nltk_score_lag_1d'] = all_data['polarty_score_with_nltk'].shift(1)
all_data['News_sentiment_lag_1d'] = all_data['News_sentiment'].shift(1)
all_data['numOfComments_lag_1d']=all_data['numOfComments'].shift(1)

"""
different regression
"""
Y_list = ['daily_return','volatility_30_days']
sentiment_score_list = ["nltk_score_lag_1d","News_sentiment_lag_1d","textblob_score_lag_1d"]
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {}'.format(y,sentiment_score), all_data).fit().summary())
                
# only number of counts?
for y in Y_list:
    print(smf.ols('{} ~ numOfComments_lag_1d'.format(y), all_data).fit().summary())

# also control number of counts?
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data).fit().summary())
