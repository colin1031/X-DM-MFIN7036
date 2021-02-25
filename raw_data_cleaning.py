# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:51:46 2021

@author: colin
"""


"""
Raw Data Cleaning
"""
path_raw_data=''

raw_data_tweets=pd.read_csv(path_raw_data+os.sep+'raw_data_tweets.csv')

#check duplicate
raw_data_tweets.drop_duplicates()

raw_data_tweets.columns
extract_columns_list_cleaning_data_use=['created_at','date','time','username','tweet','language',\
                                        'mentions','replies_count','retweets_count','likes_count','hashtags','cashtags','retweet'\
                                            ]

cleaning_data_tweets_1=raw_data_tweets[extract_columns_list_cleaning_data_use]
cleaning_data_tweets_1.columns

cleaning_data_path=r'D:\iCloudDrive\Documents\Colin\HKU\7036\X-dm\cleaning_data'
cleaning_data_tweets_1.to_csv(cleaning_data_path+os.sep+'cleaning_data_tweets_1.csv',index=False)

cleaning_data_tweets_1=pd.read_csv(cleaning_data_path+os.sep+'cleaning_data_tweets_1.csv')

cleaning_data_path=''
cleaning_data_tweets_1.to_csv(cleaning_data_path+os.sep+'cleaning_data_tweets_1.csv',index=False)
