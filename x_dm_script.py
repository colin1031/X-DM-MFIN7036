# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:12:06 2021

@author: colin
"""
"""
To download the raw data, please go to https://drive.google.com/drive/folders/1PAr0U7jk9AjHdAMOBzPB3kWlNuhs9svK?usp=sharing
"""

"""
if need:
pip install tweepy
pip uninstall twint -y
pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
pip install asyncio
pip install nest_asyncio

"""

"""
import
"""
import os
import datetime
import time
import twint
import pandas as pd
import asyncio
import nest_asyncio
import glob


"""
Setting directory
"""
os.getcwd()

path=''path)

os.getcwd()


"""
Mining tweets through Twint (Specific time range)
"""

#configuration
config = twint.Config()
config.Search = ['Xrp' or "Ripple"] #if tweet include xrp or ripple (Capital letter or not doesn't matter)
config.Lang = "en"
config.Limit = 100000000000000000000

start_date = '2020-10-01'
timeperiod = 365

start_timetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')

start_time_count = time.time()

for i in range(timeperiod):
    
    try;
        since = (start_timetime + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
        until = (start_timetime + datetime.timedelta(days = i+1)).strftime('%Y-%m-%d')

        config.Since = since
        config.Until = until

        #If we want to store in json
        # config.Store_json = True
        # config.Output = "custom_out.json"

        #If we want to store in csv
        config.Store_csv = True
        config.Output = "raw_xrp_or_ripple_twint_{}.csv".format(since)

        #Check problem: This event loop is already running exist
        loop = asyncio.get_event_loop()
        loop.is_running()

        #If True, run this command
        if loop.is_running() == True:
            nest_asyncio.apply()

        #running search
        twint.run.Search(config)
    
    except:
        i=i-1
        continue

print("--- %s seconds ---" % (time.time() - start_time_count))

"""
Read raw data (tweets through Twint with specific time range) & concate those files into one large datafile
"""
path_data=''


all_files = glob.glob(mac_path_data + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

raw_data_tweets_thro_twint = pd.concat(li, axis=0, ignore_index=True)

raw_data_tweets_thro_twint.to_csv(path_data+os.sep+'raw_data_tweets.csv')

"""
Raw Data Cleaning
"""
path_data=''

raw_data_tweets=pd.read_csv(path_data+os.sep+'raw_data_tweets.csv')

#check duplicate
raw_data_tweets.drop_duplicates()

raw_data_tweets.columns
extract_columns_list_cleaning_data_use=['created_at','date','time','username','tweet','language',\
                                        'mentions','replies_count','retweets_count','likes_count','hashtags','cashtags','retweet'\
                                            ]

cleaning_data_tweets_1=raw_data_tweets[extract_columns_list_cleaning_data_use]
cleaning_data_tweets_1.columns

"""
tweets level 
"""

"""
Extract number of followers from each user through Tweepy
likes

"""

"""
Grading tweets based on sentiments 

試試不同的打分的方法
"""

"""
Time period level timeseries (dummy variab)
"""


"""
新的dataset  based on tweest dataset
3 dims
(提及量,火熱dummy), (重要dummy), (正面/中性/負面時期)

每天的提及量 (group by) & Generating a dummy variable (1 rep heat period, 0 rep non) ?
全language mention, eng/non english, eng/chinese/etc 三種都加進去 看哪個有法


Generating a dummy variable (1 rep impactful tweet, o rep non)
Generating a dummy variable (1 rep positive emotion, 0 rep negative emotion)
Generating lag variable (t-1) (number of mention mentions ) 昨天會不會影響今天的

"""


"""
concate/merge (financial data 跟 twitter 新的dataset )
"""

""" (volatility and price change)
分析 regression
1.單變量 一個一個套
2.三變量 
3. 把其他金融變量加進去 
"""

""" (volatility and price change)
用圖分析
對比同時期
"""

"""
預測
(volatility and price change)
用machine learning 找出最好的model(regression model 的machine learning)
"""

"""
根據這些因子做個backtesting (quant trading)
"""

"""
Extra goals
"""
