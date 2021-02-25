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

# win_path=r''
# os.chdir(win_path)

mac_path=''
os.chdir(mac_path)

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
mac_path_data='/Users/colinko/Documents/Colin/HKU/7036/X-dm/raw_data'
# win_path_data=r'D:\iCloudDrive\Documents\Colin\HKU\7036\X-dm\raw_data'

all_files = glob.glob(mac_path_data + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

raw_data_tweets_thro_twint = pd.concat(li, axis=0, ignore_index=True)

raw_data_tweets_thro_twint.to_csv(mac_path_data+os.sep+'raw_data_tweets.csv')

"""
Raw Data Cleaning
"""
mac_path_data='/Users/colinko/Documents/Colin/HKU/7036/X-dm/raw_data'
# win_path_data=r'D:\iCloudDrive\Documents\Colin\HKU\7036\X-dm\raw_data'

raw_data_tweets=pd.read_csv(mac_path_data+os.sep+'raw_data_tweets.csv')

raw_data_tweets.columns
extract_columns_list_cleaning_data_use=['created_at','date','time','username','tweet','language',\
                                        'mentions','replies_count','retweets_count','likes_count','hashtags','cashtags','retweet'\
                                            ]
cleaning_data_tweets=raw_data_tweets[extract_columns_list_cleaning_data_use]
cleaning_data_tweets.columns

"""
Extract number of followers from each user
"""
