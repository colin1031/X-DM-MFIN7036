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

Y_list = ['daily_return','volatility_30_days']
sentiment_score_list = ["nltk_score_lag_1d","News_sentiment_lag_1d","textblob_score_lag_1d"]

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
"""
Text directly apply machine learning to predict
"""                                      
"""
DateFrame prepare for (text to machine learning)
"""
xrp_data = pd.read_csv('final_data.csv')[['Date', 'daily_return', 'volatility_30_days']].reset_index()
lsa_data = pd.read_csv('lsa_data.csv', index_col = 0).reset_index()

lsa_ml_data = pd.merge(xrp_data, lsa_data, on = 'index').drop(columns = 'index') 
                      
"""
machine learning apply
"""
"""
prepare ml training and testing data
"""                    
# Saving feature names for later use
# all sample start from 2020-03-02
lsa_features = lsa_ml_data.iloc[:,3:].shift(1) 

# Convert to numpy array
lsa_features = np.array(lsa_features)

# Labels are the values we want to predict
lsa_labels_ret = np.array(lsa_ml_data['daily_return']) # 'daily_return'
lsa_labels_vol = np.array(lsa_ml_data['volatility_30_days']) # 'volatility_30_days'


# split last 35 days for test sample
lsa_train_features = lsa_features[1:-35]
lsa_train_labels_ret = lsa_labels_ret[1:-35]
lsa_train_labels_vol = lsa_labels_vol[1:-35]

lsa_test_features = lsa_features[-35:]
lsa_test_labels_ret = lsa_labels_ret[-35:]
lsa_test_labels_vol = lsa_labels_vol[-35:]
      
"""
random forest
"""
 
# Instantiate model with 100 decision trees
lsa_rf_ret = RandomForestRegressor(n_estimators = 100, random_state = 10)
lsa_rf_vol = RandomForestRegressor(n_estimators = 100, random_state = 10)
# Train the model on training data
lsa_rf_ret.fit(lsa_train_features, lsa_train_labels_ret)
lsa_rf_vol.fit(lsa_train_features, lsa_train_labels_vol)

# Use the forest's predict method on the test data
lsa_rf_predictions_ret = lsa_rf_ret.predict(lsa_test_features)
lsa_rf_predictions_vol = lsa_rf_vol.predict(lsa_test_features)

# Calculate the mean squared errors testing
RF_mse_testing_ret = np.square(np.subtract(lsa_rf_predictions_ret, lsa_test_labels_ret)).mean()
RF_mse_testing_vol = np.square(np.subtract(lsa_rf_predictions_vol, lsa_test_labels_vol)).mean()

"""
SVM (SVR)
"""
 
# kernel = Radial basis function kernel, we can also set it as linear/ploy/others
lsa_svr_ret = SVR(kernel='rbf', epsilon=0.05) 
lsa_svr_vol = SVR(kernel='rbf', epsilon=0.05) 


lsa_svr_ret.fit(lsa_train_features, lsa_train_labels_ret)
lsa_svr_vol.fit(lsa_train_features, lsa_train_labels_vol)


lsa_svr_rf_predictions_ret = lsa_svr_ret.predict(lsa_test_features)
lsa_svr_rf_predictions_vol = lsa_svr_vol.predict(lsa_test_features)

SVR_mse_testing_ret = np.square(np.subtract(lsa_svr_rf_predictions_ret, lsa_test_labels_ret)).mean()
SVR_mse_testing_vol = np.square(np.subtract(lsa_svr_rf_predictions_vol, lsa_test_labels_vol)).mean()
    
"""
Lasso regression (machine learning)
"""
                      
lsa_lasso_ret = linear_model.Lasso(alpha=0.1)
lsa_lasso_vol = linear_model.Lasso(alpha=0.1)

lsa_lasso_ret.fit(lsa_train_features, lsa_train_labels_ret)
lsa_lasso_vol.fit(lsa_train_features, lsa_train_labels_vol)

lsa_lasso_predictions_ret=lsa_lasso_ret.predict(lsa_test_features)
lsa_lasso_predictions_vol=lsa_lasso_vol.predict(lsa_test_features)

lasso_mse_testing_ret = np.square(np.subtract(lsa_lasso_predictions_ret, lsa_test_labels_ret)).mean()
lasso_mse_testing_vol = np.square(np.subtract(lsa_lasso_predictions_vol, lsa_test_labels_vol)).mean()
                            
"""
save those results
"""
mse_result_texttoY = {'text_Y RF ret/vol':[RF_mse_testing_ret,RF_mse_testing_vol],
              'text_Y SVR ret/vol':[SVR_mse_testing_ret,SVR_mse_testing_vol],
              'text_Y lasso ret/vol':[lasso_mse_testing_ret,lasso_mse_testing_vol]}
"""
compare which model is the best in predict return / 30 days volatility
"""
# sentiment to Y (testing MSE)
dict_all_mse_sentiment_y = {}
for dict_mse in result_list:
    dict_all_mse_sentiment_y.update(dict_mse)
series_all_mse_sentiment_y = pd.Series(dict_all_mse_sentiment_y).sort_values(axis='index')
df_vol_all_mse_sentiment_y = pd.DataFrame({'volatility_30_days': series_all_mse_sentiment_y[:10]})
df_return_all_mse_sentiment_y = pd.DataFrame({"daily_return": series_all_mse_sentiment_y[10:]})

# text to Y
mse_result_texttoY_df = pd.DataFrame.from_dict(mse_result_texttoY).T
mse_result_texttoY_df = mse_result_texttoY_df.rename(columns = {0: "daily_return", 1: "volatility_30_days"})
                       
# easy for compare
testing_mse_daily_return_compare = pd.concat([mse_result_texttoY_df,df_return_all_mse_sentiment_y], join = 'inner')
testing_mse_daily_return_compare = testing_mse_daily_return_compare.sort_values(by = 'daily_return')

testing_mse_volatility_30_days_compare = pd.concat([mse_result_texttoY_df,df_vol_all_mse_sentiment_y],join = 'inner')
testing_mse_volatility_30_days_compare = testing_mse_volatility_30_days_compare.sort_values(by = 'volatility_30_days')

print(testing_mse_daily_return_compare)
print(testing_mse_volatility_30_days_compare)

# best model(by smallest testing mse)
# In predicting daily return
print(testing_mse_daily_return_compare.head(1))
# plot testing actual y and predict y
y=Y_list[0]
labels = np.array(label_all_data_ml_sentiment_Y['{}'.format(y)])
train_labels = labels[:-35]
test_labels = labels[-35:]
rf = RandomForestRegressor(n_estimators = 100, random_state = 10)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)

predictions_df = pd.concat([pd.DataFrame(predictions), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
predictions_df = predictions_df.drop(columns = 'index')
predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
predictions_df=predictions_df.set_index("Date")

test_labels_df = pd.concat([pd.DataFrame(test_labels), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
test_labels_df = test_labels_df.drop(columns = 'index')
test_labels_df['Date'] = pd.to_datetime(test_labels_df['Date'])
test_labels_df = test_labels_df.set_index("Date")

plt.figure() 
plt.subplots(1,1)  
plt.xticks(rotation=45)  
plt.rcParams["figure.figsize"] = (10.5, 6)
plt.title('daily_return_best_prediction_testing_result')
plt.plot(predictions_df, label = "RF_predict_y_testing_set")
plt.plot(test_labels_df, label = "RF_actual_y_testing_set")
plt.legend()
plt.show()

                                      
# In predicting volatility_30_days
print(testing_mse_volatility_30_days_compare.head(1))
# plot testing actual y and predict y
predictions_RF_text_to_y_df = pd.concat([pd.DataFrame(lsa_rf_predictions_vol), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
predictions_RF_text_to_y_df = predictions_RF_text_to_y_df.drop(columns = 'index')
predictions_RF_text_to_y_df['Date'] = pd.to_datetime(predictions_RF_text_to_y_df['Date'])
predictions_RF_text_to_y_df = predictions_RF_text_to_y_df.set_index("Date")

test_labels_RF_text_to_y_df = pd.concat([pd.DataFrame(lsa_test_labels_vol), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
test_labels_RF_text_to_y_df = test_labels_RF_text_to_y_df.drop(columns = 'index')
test_labels_RF_text_to_y_df['Date'] = pd.to_datetime(test_labels_RF_text_to_y_df['Date'])
test_labels_RF_text_to_y_df = test_labels_RF_text_to_y_df.set_index("Date")

plt.figure()   
plt.subplots(1,1)   
plt.xticks(rotation=45)  
plt.rcParams["figure.figsize"] = (10.5, 6)
plt.title('volatility_30_days_best_prediction_testing_result')
plt.plot(predictions_RF_text_to_y_df, label = "RF_predict_y_testing_set")
plt.plot(test_labels_RF_text_to_y_df, label = "RF_actual_y_testing_set")
plt.legend()
plt.show()

                       

"""
After we find out the best model (from sentiment to return/30 days volatility)
use the full dataset to produce final model and plot
"""
# In daily return
whole_features = features[:]
whole_labels = labels[:]
rf = RandomForestRegressor(n_estimators = 100, random_state = 10)
rf.fit(whole_features, whole_labels)
predictions = rf.predict(whole_features)

predictions_df = pd.concat([pd.DataFrame(predictions), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
predictions_df = predictions_df.drop(columns = 'index')
predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
predictions_df = predictions_df.set_index("Date")

whole_labels_df = pd.concat([pd.DataFrame(whole_labels), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
whole_labels_df = whole_labels_df.drop(columns = 'index')
whole_labels_df['Date'] = pd.to_datetime(whole_labels_df['Date'])
whole_labels_df = whole_labels_df.set_index("Date")

plt.figure()
plt.subplots(1,1)   
plt.xticks(rotation=45)  
plt.rcParams["figure.figsize"] = (10.5, 6)
plt.title('daily_return_final_model_result')
plt.plot(predictions_df, label = "RF_predict_y_whole_set")
plt.plot(whole_labels_df, label = "RF_actual_y_whole_set")
plt.legend()
plt.show()

# In volatility_30_days
whole_lsa_rf_features = lsa_features[1:]
whole_lsa_rf_labels_vol = lsa_labels_vol[1:]

rf_vol = RandomForestRegressor(n_estimators = 100, random_state = 10)
rf_vol.fit(whole_lsa_rf_features, whole_lsa_rf_labels_vol)
lsa_rf_whole_predictions_vol = rf_vol.predict(whole_lsa_rf_features)


predictions_RF_text_to_y_df = pd.concat([pd.DataFrame(lsa_rf_whole_predictions_vol), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
predictions_RF_text_to_y_df = predictions_RF_text_to_y_df.drop(columns = 'index')
predictions_RF_text_to_y_df['Date'] = pd.to_datetime(predictions_RF_text_to_y_df['Date'])
predictions_RF_text_to_y_df = predictions_RF_text_to_y_df.set_index("Date")

whole_labels_RF_text_to_y_df = pd.concat([pd.DataFrame(whole_lsa_rf_labels_vol), all_data_ml_sentiment_Y['Date'].iloc[-35:].reset_index()], axis = 1)
whole_labels_RF_text_to_y_df = whole_labels_RF_text_to_y_df.drop(columns = 'index')
whole_labels_RF_text_to_y_df['Date'] = pd.to_datetime(whole_labels_RF_text_to_y_df['Date'])
whole_labels_RF_text_to_y_df = whole_labels_RF_text_to_y_df.set_index("Date")

plt.figure()
plt.subplots(1,1) 
plt.xticks(rotation=45)  
plt.rcParams["figure.figsize"] = (10.5, 6)
plt.title('volatility_30_days_final_model_result')
plt.plot(predictions_RF_text_to_y_df, label = "RF_predict_y_whole_set")
plt.plot(whole_labels_RF_text_to_y_df, label = "RF_actual_y_whole_set")
plt.legend()
plt.show()
