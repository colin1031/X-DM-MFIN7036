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
import all the module and package we need
"""
import os
import datetime
from datetime import datetime as dt
import statsmodels.formula.api as smf
import time
import twint
import pandas as pd
import asyncio
import nest_asyncio
import glob
import nltk
import textstat
import openpyxl
import statsmodels.formula.api as smf
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from numpy import nan
from wordcloud import WordCloud
from datetime import datetime as dt
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

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
Mining tweets through Twint (Specific time range) 
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
Sentiment Vairable (Sun Yi) Counting mentions variable  (numOfSentence)
"""
try_df = cleaning_data_tweets_1.drop(['Unnamed: 0'],axis =1)

# split data into different weeks
file_dir = './weekly/week_{}_data.pickle'
for i in range(-1,364,7):
    temp_df = try_df[(try_df['day_adjust']>=i) & (try_df['day_adjust']<=i+7)]
    temp_df.to_pickle(file_dir.format(int((i+8)/7)),index = None)

# time processing
def adjust_day(mm):
    date_T = []
    for row_index, row in mm.iterrows():
        d = row.day_x
        h = row.date_time.hour
        if h < 8:
            d -= 1
        date_T.append(d)   
        print(row_index)
        
    mm = mm.assign(day_adjust = date_T)
    return mm

# clean the stop words & the syntax
def cleaning(temp_str):  
    sentence = temp_str.replace("."," ").replace("“","").replace('"',"").replace("'","").replace(",","").replace('‘',"'").replace("’s","").replace("•","").replace("-","").replace("#","").replace("+","").replace("●","").replace("https://t","")
    wordlist = sentence.lower().split()   # user lower case
    
    #nltk.download('stopwords')
    stopword = set(stopwords.words('english'))    
    filtered_sentence = [] 
    for w in wordlist: 
        if w not in stopword: 
            filtered_sentence.append(w) 
    return filtered_sentence

def stemming(filtered_sentence):
   # nltk.download('wordnet')
    lemma_word = []                              # stemming
    wordnet_lemmatizer = WordNetLemmatizer()

    for w in filtered_sentence:
        word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
        word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
        word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
        lemma_word.append(word3)
        
    return lemma_word    
        
def gen_worddict(temp_df):
    temp_str = ''
    for string in temp_df.tweet:
        temp_str = temp_str + '' + string
        
    # cleaning
    filtered_sentence = cleaning(temp_str)
    # stemming
    lemma_word = stemming(filtered_sentence)
    
    # worddict is what we need after all cleaning
    worddict=dict()                               #  clean some left words
    
    for word in lemma_word:
        worddict[word]=worddict.get(word,0)+1
    
    return worddict

# use dictionary
workbook = openpyxl.load_workbook("./LoughranMcDonald_SentimentWordLists_2018.xlsx") 
Negative = workbook['Negative']
Positive = workbook['Positive']
Uncertainty = workbook['Uncertainty']

def to_list(sheet):       # get the words in SentimentWordLists
    listy = list()
    for row in sheet.rows:
        for cell in row:
            listy.append(cell.value)
    return listy

negative = to_list(Negative)    
positive = to_list(Positive)    
uncertainty = to_list(Uncertainty) 

# get sentiment vairables
def get_matrix(day_x,data_1):
    print('days',day_x, ' are aready adjusted!')
    temp_df = data_1[data_1['day_adjust'] == day_x]
    worddict = gen_worddict(temp_df)
    
    print('  -  worddict is aready done!')
    
    
    post = 0      # number of positive words
    neg = 0      # number of negative words
    uncer = 0    # number of uncertainty words
    for key in worddict:                 # calculate the number of words
        if key.upper() in negative:
            neg += worddict[key]
            #print(key," is negative")
        elif key.upper() in positive:
            post += worddict[key]
            #print(key," is positive")
        elif key.upper() in uncertainty:
            uncer += worddict[key]
            #print(key," is uncertainty")
            
    numOfWords = 0
    for word in worddict:
        numOfWords += worddict[word]
        
    # martix 1
    numOfSentense = len(temp_df)       # the number of sentense
    news_sentiment = (post-neg)/numOfSentense
    
    complex_word = list()
    numOfComplex = 0
    for word in worddict:
        if textstat.syllable_count(word)>=3:
            complex_word.append(word)
            numOfComplex += worddict[word]
            #print(word,worddict[word])

    Fog = 0.4*(numOfWords/numOfSentense+100*numOfComplex/numOfWords)   #Fog index
    print("  -  Fog index is ",Fog)
    
    polarty_score_with_textblob = 0
    polarty_score_with_nltk = 0

    for sentence in temp_df['tweet']:
        #using textblob
        s = TextBlob(sentence).sentiment #assign sentiment score of that sentence
        polarty_score_with_textblob += s.polarity / numOfSentense
    
        #using nltk
        sid = SentimentIntensityAnalyzer()
        polarty_score_with_nltk += sid.polarity_scores(sentence)['compound'] / numOfSentense      #assign sentiment score of that sentence #only extract the compound score
    
    return worddict,post,neg,uncer, numOfWords,numOfSentense,news_sentiment,Fog, polarty_score_with_textblob, polarty_score_with_nltk

week_list = list(range(-1,364,7))

# process weekly data
list_1 = list()
for week in week_list:
    td = pd.read_csv(file_dir.format(int((week+8)/7)))
    day_list = list(range(week,week+7))
    for day_x in day_list:
        worddict,post,neg,uncer, numOfWords,numOfSentense,news_sentiment,Fog = get_matrix(day_x,td)
        list_1.append({ 'day':day_x,'postive': post,'negative':neg,'uncertainty':uncer,'numOfWords':numOfWords,'numOfComments':numOfSentense, 'News_sentiment': news_sentiment,'Fog index':Fog })

kk = pd.DataFrame(list_1)
kk.to_pickle('final_data_year.pickle')

# Visualization of Sentiment Scores

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, ax = plt.subplots(1, 1)

data = pd.read_csv('./final_data.csv')
data2 = pd.read_csv('./Cleaned Ripple Financial Data.csv')

names = data['date']
x = range(len(names))
 
y_1 = data['News_sentiment']
y_2 = data['polarty_score_with_nltk']
y_3 = data['polarty_score_with_textblob']
y_4 = data2['daily return']
 
plt.plot(x, y_1, color = 'red', marker = 'D', mec='r', mfc='w', linestyle = '-',lw=1  , ms=3, label = 'News Sentiment Score')
plt.plot(x, y_2, color = 'blue', marker = '*', mec='b', mfc='w',linestyle = '-',lw=1  , ms=3, label = 'Polarty Score With nltks')
plt.plot(x, y_3, color = 'green', marker = 'o', mec='g', mfc='w', linestyle = '-', lw=1  , ms=3, label = 'Polarty Score With textblob')
plt.plot(x, y_4, color = 'yellow', marker = 'o', mec='y', mfc='w', linestyle = '-', lw=2  , ms=4, label = 'Real Daily Return')


plt.legend() 
plt.xticks(x, names, rotation=45)
plt.ylabel("values") 
# plt.tight_layout()

tick_spacing = 10 
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing ))

plt.show()

#senteniment score with nltk and textblob

# cleaning_data_tweets_sentiment=cleaning_data_tweets_1
# cleaning_data_tweets_sentiment_en_only=cleaning_data_tweets_sentiment[cleaning_data_tweets_sentiment['language']=='en']
# stop_words = set(stopwords.words('english')) #get the stopword set
# tokenized_and_stopword_removed_and_lowercased_sentences_list=[]
# tokenizer = nltk.RegexpTokenizer(r"\w+") #using RegexpTokenizer to tokenize and remove all punctuation marks

# #remove stopword and lowercase
# for sentence in cleaning_data_tweets_sentiment['tweet']:
#     try:
#         word_tokens = tokenizer.tokenize(sentence)
#         tokenized_and_stopword_removed_and_lowercased_sentences_list.append([w.lower() for w in word_tokens if not w in stop_words])  #lowercase all the words
#     except:
#         tokenized_and_stopword_removed_and_lowercased_sentences_list.append([nan])

# cleaning_data_tweets_sentiment['fixed_tweets'] = [' '.join(i) for i in tokenized_and_stopword_removed_and_lowercased_sentences_list]

# cleaning_data_tweets_sentiment['polarty_score_with_textblob']= [TextBlob(sentence).sentiment.polarity for sentence in cleaning_data_tweets_sentiment['fixed_tweets']]
# #nltk
# sid = SentimentIntensityAnalyzer()
# cleaning_data_tweets_sentiment['polarty_score_with_nltk'] = [sid.polarity_scores(sentence)['compound'] for sentence in cleaning_data_tweets_sentiment['fixed_tweets']]

"""
Financial dataset related (scrape, clean process, calculate daily return stuff (lyu)
"""


"""
merge financial data and sentiment variable and number of counts
"""
# read ripple financial data   
financial_data_path='/Users/colinko/Documents/Colin/HKU/7036/X-dm/financial_data'
xrp_data = pd.read_pickle(financial_data_path+os.sep+'Cleaned Ripple Financial Data.pickle').reset_index()

#this data included number of counts already
sentiment_data=pd.read_csv(cleaning_data_path+os.sep+'sentiment_variable_data.csv',parse_dates=['date']) # semmes already hv number of counts
sentiment_data.rename(columns={'date':'Date'},inplace=True)


all_data = pd.merge(xrp_data, sentiment_data,
                    on='Date')

all_data.columns
all_data['textblob_score_lag_1d'] = all_data['polarty_score_with_textblob'].shift(1)
all_data['nltk_score_lag_1d'] = all_data['polarty_score_with_nltk'].shift(1)
all_data['News_sentiment_lag_1d'] = all_data['News_sentiment'].shift(1)
all_data['numOfComments_lag_1d']=all_data['numOfComments'].shift(1)
all_data['Fog index_lag_1d']=all_data['Fog index'].shift(1)


"""
different regression
"""
Y_list=['daily_return','volatility_30_days']
sentiment_score_list=["nltk_score_lag_1d","News_sentiment_lag_1d","textblob_score_lag_1d, Fog index_lag_1d"]
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {}'.format(y,sentiment_score), all_data).fit().summary())
        
#siginifiance in volatility, but not return
        
#only number of counts?
for y in Y_list:
    print(smf.ols('{} ~ numOfComments_lag_1d'.format(y), all_data).fit().summary())

#siginifiance in volatility, but not return

#also control number of counts?
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data).fit().summary())

#Can we this regression to predict tmr return (4 day for testing)
result_list=[]
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        a=smf.ols('{} ~ {}'.format(y,sentiment_score), all_data[:-65]).fit().summary2().tables[1]
        next_y = all_data.iloc[-65:]['daily_return']
        predicted_x=a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-65:]['{}'.format(sentiment_score)]
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"{},{},MSE_testing".format(y,sentiment_score):mse_testing})

#in return, 'daily_return,nltk_score_lag_1d,MSE_testing' lowest
#in volatility,  'volatility_30_days,nltk_score_lag_1d,MSE_testing' lowest

#also control number of mentions? Multi vairable -regression model
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        a=smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data[:-65]).fit().summary2().tables[1]
        next_y = all_data.iloc[-65:]['daily_return']
        predicted_x=a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-65:]['{}'.format(sentiment_score)]+a['Coef.'].iloc[2] * all_data.iloc[-65:]["numOfComments_lag_1d"]
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"{},{},'numOfComments_lag_1d'".format(y,sentiment_score):mse_testing})

# control for number of mentions, testing mse work  better than single (return) "daily_return,textblob_score_lag_1d,'numOfComments_lag_1d'": 0.007284506071957778
# volatility_30_days does hv improvement  {'volatility_30_days,nltk_score_lag_1d,MSE_testing': 0.011329415500966729},

"""""
Machine learning (Sentiment to Y)
"""""

"""
random forest
"""
# drop for further setting features use
all_data_rondom_forest=all_data.drop(['day','Fog index','polarty_score_with_textblob','polarty_score_with_nltk',"News_sentiment","numOfComments"],axis=1)
all_data_rondom_forest.columns
# Convert to numpy array
all_data_rondom_forest=all_data_rondom_forest.dropna()
features= all_data_rondom_forest.iloc[:,3:]
features.columns
features = np.array(features)
train_features = features[:-65]
test_features = features[-65:]

for y in Y_list:
    # Labels are the values we want to predict
    labels = np.array(all_data_rondom_forest['{}'.format(y)]) #'volatility_30_days'
    # Saving feature names for later use
    train_labels = labels[:-65]
    test_labels = labels[-65:]
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 10)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    mse_testing = np.square(np.subtract(predictions,test_labels)).mean()
    #mse for machine learning
    result_list.append({"{},random_forest".format(y):mse_testing})


"""
SVM (SVR)
"""
mse_from_SVR=[]
all_data_SVR=all_data.dropna()
#初始化SVR
for y in Y_list:
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})
    
    svr.fit(all_data_SVR.iloc[:-65,3:], all_data_SVR["{}".format(y)].iloc[:-65])
    
    y_svr = svr.predict(all_data_SVR.iloc[-65:,3:])
    result_list.append({"{},SVR".format(y):np.square(np.subtract(all_data_SVR["daily_return"].iloc[-65:],y_svr)).mean()})



