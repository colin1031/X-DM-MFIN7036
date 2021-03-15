#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import all the packages and set up directory
we suggest put xdm_init.py file in your working directory
"""


from xdm_init import *

#input your working directory
set_dic('')


"""
Read the data (check point)
"""
cleaning_data_tweets_1 = pd.read_pickle('./cleaning_data_tweets_1.pickle')
cleaning_data_tweets_1.iloc[1]


"""
Sentiment Vairable (Sun Yi) Counting mentions variable  (numOfSentence)
"""
try_df = cleaning_data_tweets_1.drop(['Unnamed: 0'],axis = 1)

# need to create a new folder in the current path named "weekly"
# split data into different weeks
file_dir = './weekly/week_{}_data.pickle'
for i in range(-1,364,7):
    temp_df = try_df[(try_df['day_adjust'] >= i) & (try_df['day_adjust'] <= i+7)]
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
   #nltk.download('wordnet')
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
    worddict = dict()                               #  clean some left words
    
    for word in lemma_word:
        worddict[word] = worddict.get(word,0) + 1
    
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
    print('days', day_x, ' are aready adjusted!')
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
    
    news_sentiment = (post-neg) / numOfComments   
    print("Number of words: " + str(numOfWords))
    print("Number of useful words: " + str(numOfUsefulWords))
    print("Number of sentense: " + str(numOfComments) ) 
    print("News sentiment: " + str(news_sentiment))
    
    complex_word = list()
    numOfComplex = 0
    for word in worddict:
        if textstat.syllable_count(word) >= 3:
            complex_word.append(word)
            numOfComplex += worddict[word]
            #print(word,worddict[word])

    Fog = 0.4*(numOfWords / numOfComments + 100*numOfComplex / numOfWords)   # Fog index
    print("  -  Fog index is ", Fog)
    

    polarty_score_with_textblob = 0
    polarty_score_with_nltk = 0

    for sentence in temp_df['tweet']:
        # using textblob
        s = TextBlob(sentence).sentiment  # assign sentiment score of that sentence
        polarty_score_with_textblob += s.polarity / numOfComments
        print('Days: '+ str(day_x))
        print('Polarty score with textblob: '+ str(polarty_score_with_textblob))
        # using nltk
        sid = SentimentIntensityAnalyzer()
        polarty_score_with_nltk += sid.polarity_scores(sentence)['compound'] / numOfComments      # assign sentiment score of that sentence #only extract the compound score
        
        print('Polarty score with nltk: '+ str(polarty_score_with_nltk))


    return worddict, post, neg, uncer, numOfWords, numOfUsefulWords, numOfComments, news_sentiment, strDay, Fog, polarty_score_with_textblob, polarty_score_with_nltk

start_date = '2020-03-01'
end_date = '2021-03-04'

week_list = list(range(-1,364,7))

# do not recommend you to do sentiment scoring again since there are about millions of comments per day
# and it cost us more than 10 hours to get the results
# process weekly data
list_1 = list()
listTotal = list()
for week in week_list:
    td = pd.read_pickle(file_dir.format(int((week+8)/7)))
    day_list = list(range(week,week+7))
    print(range(week,week+7))
    for day_x in day_list:
        worddict, post, neg, uncer, numOfWords, numOfUsefulWords, numOfComments, news_sentiment, strDay, Fog, polarty_score_with_textblob, polarty_score_with_nltk = get_matrix(day_x, td)
        listTotal.append(strDay)
        
        list_1.append({ 'day':day_x, 'postive': post, 'negative':neg, 'uncertainty':uncer, 'numOfUsefulWords': numOfUsefulWords, 'numOfWords':numOfWords, 'numOfComments':numOfComments, 'News_sentiment': news_sentiment,'Fog_index':Fog, 'polarty_score_with_textblob':polarty_score_with_textblob, 'polarty_score_with_nltk': polarty_score_with_nltk}

                                           
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

daily_return.rename(columns = {'Change %':'daily_return'}, inplace = True)

daily_return['volatility_30_days'] = np.nan
# volatility for every 30 days windows
for day in range(30, len(daily_return)):
    daily_return['volatility_30_days'].iloc[day] = np.std(daily_return.daily_return.iloc[day - 30: day])
daily_return = daily_return.dropna()

"""
merge financial data and sentiment data
"""
result = pd.merge(kk, daily_return, on = ['Date'])
result.set_index(['Date'], inplace = True)
result.to_csv('./final_data.csv')  

"""
text to vector # tfidf and LSA(Dimensionality Reduction) for text to Y analysis
"""
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(listTotal) 
terms = vectorizer.get_feature_names()
print(X)
print(terms)

n_pick_topics = 1362          # reduce dimensions to 1362
lsa = TruncatedSVD(n_pick_topics)
X2 = lsa.fit_transform(X)
print(X2)  
ll = pd.DataFrame(X2)
ll.to_csv('./lsa_data.csv')      
                      
"""                      
Visualization of Sentiment Scores
"""
fig, ax = plt.subplots(1, 1)

data = pd.read_csv('final_data.csv')
date = data['Date']
x = range(len(date))
 
y_1 = data['News_sentiment']
y_2 = data['polarty_score_with_nltk']
y_3 = data['polarty_score_with_textblob']
y_4 = data['daily_return']
y_5 = data['volatility_30_days']

plt.plot(x, y_1, color = 'springgreen', mec = 'r', mfc = 'w', linestyle = '-',lw = 1  , ms = 3, label = 'News Sentiment Score')
plt.plot(x, y_2, color = 'orangered', mec = 'b', mfc = 'w',linestyle = '-',lw = 1  , ms = 3, label = 'Polarty Score With nltks')
plt.plot(x, y_3, color = 'royalblue', mec ='g', mfc ='w', linestyle = '-', lw = 1  , ms = 3, label = 'Polarty Score With textblob')
plt.plot(x, y_4, color = 'gold', mec = 'y', mfc = 'w', linestyle = '-', lw = 1.5  , ms = 4, label = 'Real Daily Return')
plt.plot(x, y_5, color = 'violet', mec = 'y', mfc = 'w', linestyle = '-', lw = 1.5  , ms = 4, label = '30-day Volatility')


plt.legend() 
plt.xticks(x, date, rotation=45)
plt.ylabel("values") 
# plt.tight_layout()

tick_spacing = 10 # change density of X axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing ))

plt.show()
