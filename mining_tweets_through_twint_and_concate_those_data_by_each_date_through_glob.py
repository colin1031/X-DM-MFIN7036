# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:45:46 2021

@author: colin
"""

import datetime
import os
import twint
import pandas as pd
import asyncio
import nest_asyncio
import glob
import time

"""
Mining tweets through Twint (Specific time range)
"""

#specific file to save those data
path_raw_data=r'D:\iCloudDrive\Documents\Colin\HKU\7036\X-dm\raw_data'
#path_raw_data='/Users/colinko/Documents/Colin/HKU/7036/X-dm/raw_data'

os.chdir(path_raw_data)


#configuration
config = twint.Config()
config.Search = ['Xrp' or "Ripple"]
config.Lang = "en"
config.Limit = 100000000000000000000

start_date = '2020-02-01'
timeperiod = 365

start_timetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')

start_time_count = time.time()

for i in range(timeperiod):
    
    try:
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
# path_raw_data='/Users/colinko/Documents/Colin/HKU/7036/X-dm/raw_data'
path_raw_data=r'D:\iCloudDrive\Documents\Colin\HKU\7036\X-dm\raw_data'

all_files = glob.glob(path_raw_data + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

raw_data_tweets_thro_twint = pd.concat(li, axis=0, ignore_index=True)

raw_data_tweets_thro_twint.to_csv(path_raw_data+os.sep+'raw_data_tweets.csv',index=False)
