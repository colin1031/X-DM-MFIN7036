# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:12:06 2021

@author: colin
"""

"""
if need:
pip install tweepy
pip install twint
"""

"""
import
"""
import os
import tweepy
import time
import twint
"""
Setting directory
"""
os.getcwd()

# win_path=r'D:\iCloudDrive\Documents\Colin\HKU\7036\X-dm'
# os.chdir(win_path)

mac_path='/Users/colinko/Documents/Colin/HKU/7036/X-dm'
os.chdir(mac_path)

os.getcwd()

access_token=''
access_token_secret=''

consumer_key=''
consumer_secret=''

"""
Verifying Credentials & test authentication
"""
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


"""
Mining Tweets (Streaming)
"""

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
    
"""
if is_async=False, control c to stop    
"""
    
"""
disconnecting background stream (if is_async=True) 
"""
mystream.disconnect()

"""
Close file after mining
"""
f.close()

"""
Read txt file
"""
f = open(mac_path+os.sep+'data-streaming-tweets.txt', "r",encoding="utf-8")
f.read()
f.close()

"""
Mining Old tweets
"""

#configuration
config = twint.Config()
config.Search = "bitcoin"
config.Lang = "en"
config.Limit = 100
config.Since = "2019–04–29"
config.Until = "2019–09–29"
config.Store_json = True
config.Output = "custom_out.json"

#running search
twint.run.Search(config)


