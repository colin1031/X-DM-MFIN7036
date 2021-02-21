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

mac_path=''
os.chdir(mac_path)

os.getcwd()


"""
Mining Old tweets through Twint
"""

#configuration
config = twint.Config()
config.Search = ['Xrp' or "Ripple"] #if tweet include xrp or ripple (Capital letter or not doesn't matter)
config.Lang = "en"
config.Limit = 100000000000000000000
config.Since = '2020-02-01'
config.Until = '2021-02-01'

#If we want to store in json
# config.Store_json = True
# config.Output = "custom_out.json"

#If we want to store in csv
config.Store_csv = True
config.Output = "xrp_or_ripple_twint.csv"

#Check problem: This event loop is already running exist
loop = asyncio.get_event_loop()
loop.is_running()

#If True, run this command
if loop.is_running() == True:
    nest_asyncio.apply()

#running search
twint.run.Search(config)
