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
Read merged data (financial data and sentiment data)
"""
# read 
all_data = pd.read_csv('./final_data.csv')

all_data.columns
all_data['textblob_score_lag_1d'] = all_data['polarty_score_with_textblob'].shift(1)
all_data['nltk_score_lag_1d'] = all_data['polarty_score_with_nltk'].shift(1)
all_data['News_sentiment_lag_1d'] = all_data['News_sentiment'].shift(1)
all_data['numOfComments_lag_1d']=all_data['numOfComments'].shift(1)

"""
different regression
"""
Y_list = ['daily_return','volatility_30_days']
sentiment_score_list = ["nltk_score_lag_1d","News_sentiment_lag_1d","textblob_score_lag_1d"]
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {}'.format(y,sentiment_score), all_data).fit().summary())
                
# only number of counts?
for y in Y_list:
    print(smf.ols('{} ~ numOfComments_lag_1d'.format(y), all_data).fit().summary())

# also control number of counts?
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        print(smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data).fit().summary())

"""
Prediction with regression (sentiment to Y)
"""
# Can we this regression to predict future return and future 30 days volatility (testing set)
# We use testing mse to compare the prediction model performance
result_list = []
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        a = smf.ols('{} ~ {}'.format(y,sentiment_score), all_data[:-35]).fit().summary2().tables[1]
        next_y = all_data.iloc[-35:]['{}'.format(y)]
        predicted_x = a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-35:]['{}'.format(sentiment_score)]
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"sentiment_Y, {}, {}, mse_testing".format(y,sentiment_score): mse_testing})

#predict with only mentions count
for y in Y_list:
        a = smf.ols('{} ~ numOfComments_lag_1d'.format(y), all_data[:-35]).fit().summary2().tables[1]
        next_y = all_data.iloc[-35:]['{}'.format(y)]
        predicted_x = a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-35:]['numOfComments_lag_1d']
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"sentiment_Y, {}, numOfComments_lag_1d, mse_testing".format(y): mse_testing})

#also control number of mentions? Multi vairable -regression model
for y in Y_list:
    for sentiment_score in sentiment_score_list:
        a = smf.ols('{} ~ {} + numOfComments_lag_1d'.format(y,sentiment_score), all_data[:-35]).fit().summary2().tables[1]
        next_y = all_data.iloc[-35:]['{}'.format(y)]
        predicted_x = a['Coef.'].iloc[0] + a['Coef.'].iloc[1] * all_data.iloc[-35:]['{}'.format(sentiment_score)]+a['Coef.'].iloc[2] * all_data.iloc[-35:]["numOfComments_lag_1d"]
        mse_testing = np.square(np.subtract(next_y,predicted_x)).mean()
        result_list.append({"sentiment_Y,{},{},'numOfComments_lag_1d,mse_testing'".format(y,sentiment_score):mse_testing})

"""""
Machine learning (Sentiment to Y)
"""""
# drop for further setting features use
all_data_ml_sentiment_Y = all_data.drop(['Fog_index','polarty_score_with_textblob','polarty_score_with_nltk',"News_sentiment","numOfComments"],axis = 1).dropna()
label_all_data_ml_sentiment_Y = pd.concat([all_data_ml_sentiment_Y.pop(x) for x in ['daily_return', 'volatility_30_days']],axis = 1)

"""
random forest
"""

# Convert to numpy array
features = all_data_ml_sentiment_Y.iloc[:,1:]
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
    # Calculate the mean squared errors testing
    mse_testing = np.square(np.subtract(predictions, test_labels)).mean()
    #mse for machine learning
    result_list.append({"sentiment_Y, {}, random_forest, mse_testing".format(y): mse_testing})
                      
"""
SVM (SVR)
"""                      
svr_training_x = all_data_ml_sentiment_Y.iloc[:-35,1:]
svr_testing_x = all_data_ml_sentiment_Y.iloc[-35:,1:]
                      
for y in Y_list:
    svr = SVR(kernel = 'rbf', epsilon = 0.05) #kernel= Radial basis function kernel, we can also set it as linear/ploy/others
    svr_training_y = label_all_data_ml_sentiment_Y["{}".format(y)].iloc[:-35]
    svr.fit(svr_training_x, svr_training_y)
    
    y_svr = svr.predict(svr_testing_x)
    svr_testing_y = label_all_data_ml_sentiment_Y["{}".format(y)].iloc[-35:]
    result_list.append({"sentiment_Y, {}, SVR, mse_testing".format(y):np.square(np.subtract(svr_testing_y,y_svr)).mean()})
                       
"""
Lasso regression (machine learning)
"""
lasso_training_x = all_data_ml_sentiment_Y.iloc[:-35,3:]
lasso_testing_x = all_data_ml_sentiment_Y.iloc[-35:,3:]
for y in Y_list:
    clf = linear_model.Lasso(alpha  =0.1)
    lasso_training_y = label_all_data_ml_sentiment_Y["{}".format(y)].iloc[:-35]
    clf.fit(lasso_training_x, lasso_training_y)
    y_lasso = clf.predict(lasso_testing_x)
                      
    lasso_testing_y = label_all_data_ml_sentiment_Y["{}".format(y)].iloc[-35:]
    result_list.append({"sentiment_Y,{},lasso,mse_testing".format(y): np.square(np.subtract(lasso_testing_y,y_lasso)).mean()})
                       

