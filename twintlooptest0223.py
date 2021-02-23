#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:00:33 2021

@author: zhongyifeng
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:12:06 2021

@author: colin
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

"""
Setting directory
"""
os.getcwd()

# win_path=r''
# os.chdir(win_path)

# mac_path=''
# os.chdir(mac_path)

# os.getcwd()


"""
Mining tweets through Twint (Specific time range)
"""

#configuration
config = twint.Config()
config.Search = ['Xrp' or "Ripple"] #if tweet include xrp or ripple (Capital letter or not doesn't matter)
config.Lang = "en"
config.Limit = 1000000

start_date = '2020-10-01'
timeperiod = 20



for i in range(timeperiod):
   
    start_timetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    since = (start_timetime + datetime.timedelta(days = i)).strftime('%Y-%m-%d')
    until = (start_timetime + datetime.timedelta(days = i+1)).strftime('%Y-%m-%d')
    config.Since = since
    config.Until = until
    
    
    #If we want to store in json
    # config.Store_json = True
    # config.Output = "custom_out.json"
    
    #If we want to store in csv
    config.Store_csv = True
    config.Output = f"xrp_or_ripple_twint{since}.csv"
    
    #Check problem: This event loop is already running exist
    loop = asyncio.get_event_loop()
    loop.is_running()
    
    #If True, run this command
    if loop.is_running() == True:
        nest_asyncio.apply()
    
    #running search
    twint.run.Search(config)
