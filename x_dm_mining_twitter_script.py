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
import tweepy
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
config.Search = ['Xrp' or "Ripple"]
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


"""
If we want to stream tweets through tweepy
"""
#Keys entries
access_token=''
access_token_secret=''

consumer_key=''
consumer_secret=''

#Verifying Credentials & test authentication
# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)
# test authentication
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

#Mining Tweets (Streaming)
class listener(tweepy.StreamListener):
    def on_status(self, status):
        with open('data-streaming-tweets.txt', 'a',encoding="utf-8") as f:
            if hasattr(status, 'retweeted_status'):
                f.write('retweeted'+" : "+\
                    status.user.screen_name + ' : ' + \
                    str(status.user.followers_count) + ' : ' + \
                    str(status.created_at) + ' : ' + \
                    status.text + '\n') 
            else:
                f.write('not_retweet'+" : "+\
                    status.user.screen_name + ' : ' + \
                    str(status.user.followers_count) + ' : ' + \
                    str(status.created_at) + ' : ' + \
                    status.text + '\n')
                
        return True
    
    def on_error(self, status_code):
        print(status_code)
        return True
    
    def on_limit(self,status):
        print ("Rate Limit Exceeded, Sleep for 2 Mins")
        time.sleep(2 * 60)
        return True

mystream = \
    tweepy.Stream(
        auth=api.auth,
        listener=listener())
    
mystream.filter(track=['XRP','Xrp','xrp','RIPPLE','Ripple','ripple'],is_async=True)

#is_async=True #background streaming ,will not run in console
    
#if we set is_async=False, control c to stop    

#disconnecting background stream (if is_async=True) 
mystream.disconnect()

#Close file after mining
f.close()

#Read txt file
f = open(mac_path+os.sep+'data-streaming-tweets.txt', "r",encoding="utf-8")
f.read()
f.close()
