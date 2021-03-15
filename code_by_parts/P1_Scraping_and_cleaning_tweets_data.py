"""
To download all the data, please go to:
https://drive.google.com/drive/folders/1PAr0U7jk9AjHdAMOBzPB3kWlNuhs9svK?usp=sharing
"""

"""
If encounter trouble when using twint:
pip uninstall twint -y
pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

"""

"""
import all the packages and set up directory
we suggest put xdm_init.py file in your working directory
"""



from xdm_init import *

#input your working directory
set_dic('')


"""
Scraping tweets through Twint (Specific time range) 
This part will generate two file: raw_data_tweets.pickle and cleaning_data_tweets_1.pickle 
we suggest not to run this part if you have tweet text data needed for this project as running this will take a long time
"""

#twint will save data to current directory

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


# Read raw data (tweets through Twint with specific time range) & concate those files into one large datafile

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col = None, header = 0)
    li.append(df)

raw_data_tweets_thro_twint = pd.concat(li, axis = 0, ignore_index = True)

raw_data_tweets_thro_twint.to_pickle('./raw_data_tweets.pickle')


# Raw Data Cleaning and preprocessing

raw_data_tweets = pd.read_pickle('./raw_data_tweets.pickle')

#check duplicates and drop duplicates
raw_data_tweets.drop_duplicates(inplace = True)

#fix time in order to match the finacial data timezone
def switch_tz(time, t = 8):
    return datetime.datetime.strptime(time[:19], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours = t)

raw_data_tweets['datetime'] = raw_data_tweets.created_at.apply(lambda x:switch_tz(x))

raw_data_tweets['date']= raw_data_tweets.datetime.apply(lambda x:x.date())

#check columns
raw_data_tweets.columns

extract_columns_list_cleaning_data_use = ['date','user_id','tweet','language']

cleaning_data_tweets_1 = raw_data_tweets[extract_columns_list_cleaning_data_use]
cleaning_data_tweets_1.columns
#from 36 columns (raw data) drop to 4 columns now

#drop_duplicates again after fixed the date issue
cleaning_data_tweets_1.drop_duplicates(inplace = True)

#save file for further step
cleaning_data_tweets_1.to_pickle('./cleaning_data_tweets_1.pickle')
