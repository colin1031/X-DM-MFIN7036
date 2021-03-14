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
