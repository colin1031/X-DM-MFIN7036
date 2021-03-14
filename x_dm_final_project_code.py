# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:12:06 2021

@author: colin
"""
"""
To download the data, please go to:
https://drive.google.com/drive/folders/1PAr0U7jk9AjHdAMOBzPB3kWlNuhs9svK?usp=sharing
"""

"""
If having trouble when using twint:
pip uninstall twint -y
pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

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
from sklearn.model_selection import learning_curve
from sklearn import linear_model


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

#check duplicates and drop duplicates
raw_data_tweets.drop_duplicates(inplace=True)

#fix time in order to match the finacial data timezone
def switch_tz(time, t=8):
    return datetime.datetime.strptime(time[:19], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours = t)

raw_data_tweets['datetime'] = raw_data_tweets.created_at.apply(lambda x:switch_tz(x))

raw_data_tweets['date']= raw_data_tweets.datetime.apply(lambda x:x.date())

#check columns
raw_data_tweets.columns

extract_columns_list_cleaning_data_use=['date','user_id','tweet','language']

cleaning_data_tweets_1=raw_data_tweets[extract_columns_list_cleaning_data_use]
cleaning_data_tweets_1.columns
#from 36 columns (raw data) drop to 4 columns now

#drop_duplicates again after fixed the date issue
cleaning_data_tweets_1.drop_duplicates(inplace=True)

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
    emp_df = temp_df.drop(temp_df[temp_df['language']!="en"].index)
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
    adjust_day(data_1)
    print('days',day_x, ' are aready adjusted!')
    print(data_1['day_adjust'])
    temp_df = data_1[data_1['day_adjust'] == day_x]
    print(temp_df)
    worddict = gen_worddict(temp_df)
    
    print('  -  worddict is aready done!')
    
    
    post = 0      # number of positive words
    neg = 0      # number of negative words
    uncer = 0    # number of uncertainty words
    strDay = str()
    for key in worddict:                 # calculate the number of words
        if key.upper() in negative:
            neg += worddict[key]
            
            strDay = strDay +  " " + key
            #print(key," is negative")
        elif key.upper() in positive:
            post += worddict[key]
            strDay = strDay +  " " + key
            #print(key," is positive")
        elif key.upper() in uncertainty:
            uncer += worddict[key]
            strDay = strDay +  " " + key
            #print(key," is uncertainty")
        #else:
            #print("sorry,this word is not work!——",key)
    
    numOfUsefulWords = neg + post + uncer 
       
    numOfWords = 0
    for word in worddict:
        numOfWords += worddict[word]
   
    numOfComments = len(temp_df) 
    
    news_sentiment = (post-neg)/numOfComments   
    print("Number of words: " + str(numOfWords))
    print("Number of useful words: " + str(numOfUsefulWords))
    print("Number of sentense: " + str(numOfComments) ) 
    print("News sentiment: " + str(news_sentiment))
    
    complex_word = list()
    numOfComplex = 0
    for word in worddict:
        if textstat.syllable_count(word)>=3:
            complex_word.append(word)
            numOfComplex += worddict[word]
            #print(word,worddict[word])

    Fog = 0.4*(numOfWords/numOfComments+100*numOfComplex/numOfWords)   #Fog index
    print("  -  Fog index is ",Fog)
    

    polarty_score_with_textblob = 0
    polarty_score_with_nltk = 0

    for sentence in temp_df['tweet']:
        #using textblob
        s = TextBlob(sentence).sentiment #assign sentiment score of that sentence
        polarty_score_with_textblob += s.polarity / numOfComments
        print('Days: '+ str(day_x))
        print('Polarty score with textblob: '+ str(polarty_score_with_textblob))
        #using nltk
        sid = SentimentIntensityAnalyzer()
        polarty_score_with_nltk += sid.polarity_scores(sentence)['compound'] / numOfComments      #assign sentiment score of that sentence #only extract the compound score
        
        print('Polarty score with nltk: '+ str(polarty_score_with_nltk))


    return worddict, post,neg,uncer, numOfWords, numOfUsefulWords, numOfComments, news_sentiment, strDay, Fog, polarty_score_with_textblob, polarty_score_with_nltk

start_date = '2020-03-01'
end_date = '2021-03-04'

week_list = list(range(-1,364,7))

# process weekly data
list_1 = list()
listTotal = list()
for week in week_list:
    td = pd.read_pickle(file_dir.format(int((week+8)/7)))
    day_list = list(range(week,week+7))
    print(range(week,week+7))
    for day_x in day_list:
        worddict,post,neg,uncer, numOfWords,numOfUsefulWords,numOfComments,news_sentiment,strDay, Fog, polarty_score_with_textblob, polarty_score_with_nltk = get_matrix(day_x,td)
        listTotal.append(strDay)
        
        list_1.append({ 'day':day_x,'postive': post,'negative':neg,'uncertainty':uncer,'numOfUsefulWords': numOfUsefulWords,'numOfWords':numOfWords,'numOfComments':numOfComments, 'News_sentiment': news_sentiment,'Fog_index':Fog, 'polarty_score_with_textblob':polarty_score_with_textblob, 'polarty_score_with_nltk': polarty_score_with_nltk}

# tfidf and LSA
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(listTotal) 
terms = vectorizer.get_feature_names()
print(terms)

n_pick_topics = 1362          # 设定主题数为1362，也就是将维度降到了1362
lsa = TruncatedSVD(n_pick_topics)
X2 = lsa.fit_transform(X)
print(X2)  
ll = pd.DataFrame(X2)
ll.to_csv('./lsa_data.csv')  
                                           
# add back date information                     
dates = pd.date_range(start_date,end_date).strftime("%Y-%m-%d").to_list()

kk = pd.DataFrame(list_1)
kk.drop(kk.columns[0], axis = 1, inplace = True)
kk['Date'] = dates
kk.set_index(['Date'], inplace=True)


"""
financial dataset related (calculate 30 days volatility, etc.)
"""

# merge financial data
ripple = pd.read_csv('XRP_USD Historical Data.csv')

daily_return = pd.DataFrame(ripple[['Date','Change %']])

daily_return.rename(columns={'Change %':'daily_return'}, inplace=True)

daily_return['volatility_30_days'] = np.nan
# volatility for every 30 days windows
for day in range(30, len(daily_return)):
    daily_return['volatility_30_days'].iloc[day] = np.std(daily_return.daily_return.iloc[day-30:day])
daily_return = daily_return.dropna()

"""
merge financial data and sentiment data
"""
result = pd.merge(kk, daily_return, on=['Date'])
result.set_index(['Date'], inplace=True)
result.to_csv('./final_data.csv')  
                      
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


"""
Read merged data (financial data and sentiment data)
"""
#read 
all_data = pd.read_csv('./final_data.csv')

all_data.columns
all_data['textblob_score_lag_1d'] = all_data['polarty_score_with_textblob'].shift(1)
all_data['nltk_score_lag_1d'] = all_data['polarty_score_with_nltk'].shift(1)
all_data['News_sentiment_lag_1d'] = all_data['News_sentiment'].shift(1)
all_data['numOfComments_lag_1d']=all_data['numOfComments'].shift(1)
all_data['Fog_index_lag_1d']=all_data['Fog_index'].shift(1)

"""
different regression
"""
Y_list=['daily_return','volatility_30_days']
sentiment_score_list=["nltk_score_lag_1d","News_sentiment_lag_1d","textblob_score_lag_1d","Fog_index_lag_1d"]
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {}'.format(y,sentiment_score), all_data).fit().summary())
                
#only number of counts?
for y in Y_list:
    print(smf.ols('{} ~ numOfComments_lag_1d'.format(y), all_data).fit().summary())

#also control number of counts?
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data).fit().summary())

"""
Prediction with regression (sentiment to Y)
"""
#Can we this regression to predict future return and future 30 days volatility (testing set)
#We use testing mse to compare the prediction model performance
result_list=[]
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        a=smf.ols('{} ~ {}'.format(y,sentiment_score), all_data[:-35]).fit().summary2().tables[1]
        next_y = all_data.iloc[-35:]['{}'.format(y)]
        predicted_x=a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-35:]['{}'.format(sentiment_score)]
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"sentiment_Y,{},{},mse_testing".format(y,sentiment_score):mse_testing})

#predict with only mentions count
for y in Y_list:
        a=smf.ols('{} ~ numOfComments_lag_1d'.format(y), all_data[:-35]).fit().summary2().tables[1]
        next_y = all_data.iloc[-35:]['{}'.format(y)]
        predicted_x=a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-35:]['numOfComments_lag_1d']
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"sentiment_Y,{},numOfComments_lag_1d,mse_testing".format(y):mse_testing})

#also control number of mentions? Multi vairable -regression model
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        a=smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data[:-35]).fit().summary2().tables[1]
        next_y = all_data.iloc[-35:]['{}'.format(y)]
        predicted_x=a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-35:]['{}'.format(sentiment_score)]+a['Coef.'].iloc[2] * all_data.iloc[-35:]["numOfComments_lag_1d"]
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"sentiment_Y,{},{},'numOfComments_lag_1d,mse_testing'".format(y,sentiment_score):mse_testing})

"""""
Machine learning (Sentiment to Y) #need update [:3]? based on sun yi financial data merge sentiment data
"""""
# drop for further setting features use
all_data_ml_sentiment_Y=all_data.drop(['Fog_index','polarty_score_with_textblob','polarty_score_with_nltk',"News_sentiment","numOfComments"],axis=1).dropna()
label_all_data_ml_sentiment_Y=pd.concat([all_data_ml_sentiment_Y.pop(x) for x in ['daily_return', 'volatility_30_days']],axis=1)

"""
random forest
"""

# Convert to numpy array
features= all_data_ml_sentiment_Y.iloc[:,1:]
features.columns
features = np.array(features)
train_features = features[:-35]
test_features = features[-35:]
for y in Y_list:
    # Labels are the values we want to predict
    labels = np.array(label_all_data_ml_sentiment_Y['{}'.format(y)])
    # Saving feature names for later use
    train_labels = labels[:-35]
    test_labels = labels[-35:]
    # Instantiate model with 100 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 10)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    mse_testing = np.square(np.subtract(predictions,test_labels)).mean()
    #mse for machine learning
    result_list.append({"sentiment_Y,{},random_forest,mse_testing".format(y):mse_testing})
                      
"""
SVM (SVR)
"""                      
svr_training_x=all_data_ml_sentiment_Y.iloc[:-35,1:]
svr_testing_x=all_data_ml_sentiment_Y.iloc[-35:,1:]
                      
for y in Y_list:
    svr = SVR(kernel='rbf', epsilon=0.05) #kernel= Radial basis function kernel, we can also set it as linear/ploy/others
    svr_training_y=label_all_data_ml_sentiment_Y["{}".format(y)].iloc[:-35]
    svr.fit(svr_training_x,svr_training_y)
    
    y_svr = svr.predict(svr_testing_x)
    svr_testing_y=label_all_data_ml_sentiment_Y["{}".format(y)].iloc[-35:]
    result_list.append({"sentiment_Y,{},SVR,mse_testing".format(y):np.square(np.subtract(svr_testing_y,y_svr)).mean()})
                       
"""
Lasso regression (machine learning)
"""
lasso_training_x=all_data_ml_sentiment_Y.iloc[:-35,3:]
lasso_testing_x=all_data_ml_sentiment_Y.iloc[-35:,3:]
for y in Y_list:
    clf = linear_model.Lasso(alpha=0.1)
    lasso_training_y= label_all_data_ml_sentiment_Y["{}".format(y)].iloc[:-35]
    clf.fit(lasso_training_x,lasso_training_y)
    y_lasso=clf.predict(lasso_testing_x)
                      
    lasso_testing_y= label_all_data_ml_sentiment_Y["{}".format(y)].iloc[-35:]
    result_list.append({"sentiment_Y,{},lasso,mse_testing".format(y):np.square(np.subtract(lasso_testing_y,y_lasso)).mean()})
                       
"""
Text directly apply machine learning to predict Y (fong)
"""
"""
text 轉 vector #模型tfidf lsa等 should be 2 model
"""
"""
DateFrame prepare for (text to machine learning)
"""
xrp_data = pd.read_csv('merge_data.csv')[['Date', 'daily_return', 'volatility_30_days']].reset_index()
lsa_data = pd.read_csv('lsa_data.csv', index_col=0).reset_index()

ml_data = pd.merge(xrp_data, lsa_data, on='index').drop(columns='index') 
                      
"""
machine learning apply
"""
'''
prepare ml training and testing data
'''                      
# Saving feature names for later use
#all sample start from 2020-03-02
features = ml_data.iloc[:,3:].shift(1) 

# Convert to numpy array
features = np.array(features)

# Labels are the values we want to predict
labels_ret = np.array(ml_data['daily_return']) #'daily_return'
labels_vol = np.array(ml_data['volatility_30_days']) #'volatility_30_days'


#split last 35 days for test sample
train_features = features[1:-35]
train_labels_ret = labels_ret[1:-35]
train_labels_vol = labels_vol[1:-35]

test_features = features[-35:]
test_labels_ret = labels_ret[-35:]
test_labels_vol = labels_vol[-35:]

"""
random forest
"""

# Instantiate model with 100 decision trees
rf_ret = RandomForestRegressor(n_estimators = 100, random_state = 10)
rf_vol = RandomForestRegressor(n_estimators = 100, random_state = 10)
# Train the model on training data
rf_ret.fit(train_features, train_labels_ret)
rf_vol.fit(train_features, train_labels_vol)

# Use the forest's predict method on the test data
predictions_ret = rf_ret.predict(test_features)
predictions_vol = rf_vol.predict(test_features)

# Calculate the absolute errors
RF_mse_testing_ret = np.square(np.subtract(predictions_ret,test_labels_ret)).mean()
RF_mse_testing_vol = np.square(np.subtract(predictions_vol,test_labels_vol)).mean()
print('RF_mse_ret:' + str(RF_mse_testing_ret) + '\nRF_mse_vol:' + str(RF_mse_testing_vol))

"""
SVM (SVR)
"""
    
#kernel= Radial basis function kernel, we can also set it as linear/ploy/others
svr_ret = SVR(kernel='rbf', epsilon=0.05) 
svr_vol = SVR(kernel='rbf', epsilon=0.05) 


svr_ret.fit(train_features,train_labels_ret)
svr_vol.fit(train_features,train_labels_vol)


SVR_predictions_ret = svr_ret.predict(test_features)
SVR_predictions_vol = svr_vol.predict(test_features)

SVR_mse_testing_ret = np.square(np.subtract(SVR_predictions_ret,test_labels_ret)).mean()
SVR_mse_testing_vol = np.square(np.subtract(SVR_predictions_vol,test_labels_vol)).mean()
                            
print('SVR_mse_ret:' + str(SVR_mse_testing_ret) + '\nSVR_mse_vol:' + str(SVR_mse_testing_vol))

    
"""
Lasso regression (machine learning)
"""
                      
lasso_ret = linear_model.Lasso(alpha=0.1)
lasso_vol = linear_model.Lasso(alpha=0.1)

lasso_ret.fit(train_features, train_labels_ret)
lasso_vol.fit(train_features, train_labels_vol)

lasso_predictions_ret=clf_ret.predict(test_features)
lasso_predictions_vol=clf_vol.predict(test_features)

lasso_mse_testing_ret = np.square(np.subtract(lasso_predictions_ret,test_labels_ret)).mean()
lasso_mse_testing_vol = np.square(np.subtract(lasso_predictions_vol,test_labels_vol)).mean()
                            
print('lasso_mse_ret:' + str(lasso_mse_testing_ret) + '\nclasso_mse_vol:' + str(lasso_mse_testing_vol))

mse_result_texttoY = {'text_Y RF ret/vol':[RF_mse_testing_ret,RF_mse_testing_vol],
              'text_Y SVR ret/vol':[SVR_mse_testing_ret,SVR_mse_testing_vol],
              'text_Y lasso ret/vol':[lasso_mse_testing_ret,lasso_mse_testing_vol]}
                      
"""
compare which model is the best in predict return / 30 days volatility
"""
#sentiment to Y (testing MSE)
result_list
#text to Y
mse_result_texttoY
                       
"""
After we find out the best prediction model (from sentiment to return/30 days volatility)
"""
#fit the model with whole data set

#plotting
