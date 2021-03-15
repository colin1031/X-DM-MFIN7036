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
plt.rcParams["figure.figsize"] = (10.5, 6)
plt.title('volatility_30_days_final_model_result')
plt.plot(predictions_RF_text_to_y_df, label = "RF_predict_y_whole_set")
plt.plot(whole_labels_RF_text_to_y_df, label = "RF_actual_y_whole_set")
plt.legend()
plt.show()
