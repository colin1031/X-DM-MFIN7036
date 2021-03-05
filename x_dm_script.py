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
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
from numpy import nan
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
Setting directory
"""
os.getcwd()

path=''
os.chdir(path)

os.getcwd()

"""
Setting all the path
"""

#specific file to save those data
path_raw_data=r''

cleaning_data_path=r''

"""
Mining tweets through Twint (Specific time range) #改做爬不同的標的 不同天?
"""
os.chdir(path_raw_data) #twint will save data to current directory, so we need to set directory

#configuration
config = twint.Config()
config.Search = ['Xrp' or "Ripple"] #if tweet include xrp or ripple (Capital letter or not doesn't matter)
config.Lang = "en"
config.Limit = 100000000000000000000

start_date = '2020-03-01'
timeperiod = 365+4

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
all_files = glob.glob(path_raw_data + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

raw_data_tweets_thro_twint = pd.concat(li, axis=0, ignore_index=True)

raw_data_tweets_thro_twint.to_pickle(path_raw_data+os.sep+'raw_data_tweets.pickle')

"""
Raw Data Cleaning and preprocessing
"""
raw_data_tweets=pd.read_pickle(path_raw_data+os.sep+'raw_data_tweets.pickle')

#check duplicate
raw_data_tweets.drop_duplicates()

#fix time in order to match the finacial data timezone
def switch_tz(time, t=8):
    return datetime.datetime.strptime(time[:19], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours = t)

raw_data_tweets['datetime'] = raw_data_tweets.created_at.apply(lambda x:switch_tz(x))

raw_data_tweets['date']= raw_data_tweets.datetime.apply(lambda x:x.date())

#check columns
raw_data_tweets.columns

extract_columns_list_cleaning_data_use=['date','username','tweet','language',\
                                        'mentions','replies_count','retweets_count','likes_count','hashtags','cashtags','retweet'\
                                            ]

cleaning_data_tweets_1=raw_data_tweets[extract_columns_list_cleaning_data_use]
cleaning_data_tweets_1.columns
#from 36 columns (raw data) drop to 11 columns now

#save file for further step
cleaning_data_tweets_1.to_pickle(cleaning_data_path+os.sep+'cleaning_data_tweets_1.pickle')

"""
Read the data (check point)
"""
cleaning_data_tweets_1=pd.read_pickle(cleaning_data_path+os.sep+'cleaning_data_tweets_1.pickle')
cleaning_data_tweets_1.iloc[1]

"""
making different variable
"""
"""
Counting mentions variable (Colin)
"""
cleaning_data_tweets_mention=cleaning_data_tweets_1

total_number_of_mentions_per_date=cleaning_data_tweets_mention.date.value_counts()

en_number_of_mentions=cleaning_data_tweets_mention[cleaning_data_tweets_mention['language']=='en']
en_number_of_mentions_per_day=en_number_of_mentions.groupby('date').size()
#seems like 'en' size too small

non_en_number_of_mentions_per_day=[total_number_of_mentions_per_date-en_number_of_mentions_per_day]


"""
Sentiment Vairable (Sun Yi)
#cleaning and preprocessing (each tweets)
"""

cleaning_data_tweets_sentiment=cleaning_data_tweets_1
stop_words = set(stopwords.words('english')) #get the stopword set
tokenized_and_stopword_removed_and_lowercased_sentences_list=[]
tokenizer = nltk.RegexpTokenizer(r"\w+") #using RegexpTokenizer to tokenize and remove all punctuation marks

#remove stopword and lowercase
for sentence in cleaning_data_tweets_sentiment['tweet']:
    try:
        word_tokens = tokenizer.tokenize(sentence)
        tokenized_and_stopword_removed_and_lowercased_sentences_list.append([w.lower() for w in word_tokens if not w in stop_words])  #lowercase all the words
    except:
        tokenized_and_stopword_removed_and_lowercased_sentences_list.append([nan])

cleaning_data_tweets_sentiment['fixed_tweets'] = [' '.join(i) for i in tokenized_and_stopword_removed_and_lowercased_sentences_list]

textblob_sentimentscore_list=[]
nltk_sentimentscore_list=[]

for sentence in cleaning_data_tweets_sentiment['stopword_punctuation_removed_and_lowercased_sentences']:
    #using textblob
    s = TextBlob(sentence).sentiment #assign sentiment score of that sentence
    textblob_sentimentscore_list.append(s.polarity)

    #using nltk
    sid = SentimentIntensityAnalyzer()
    nltk_sentimentscore_list.append(sid.polarity_scores(sentence)['compound']) #assign sentiment score of that sentence #only extract the compound score

#add as new columns into dataframe
cleaning_data_tweets_sentiment['polarty_score_with_textblob'] = textblob_sentimentscore_list 
cleaning_data_tweets_sentiment['polarty_score_with_nltk'] = nltk_sentimentscore_list

"""
influencer variable (Fong Fong)
Extract number of followers from each user through Tweepy
likes
"""
cleaning_data_tweets_influencer=cleaning_data_tweets_1

"""
Financial dataset related (scrape, clean process, calculate daily return stuff (lyu)
"""


"""
Merge variable into financial time series dataset & export as pickle
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
