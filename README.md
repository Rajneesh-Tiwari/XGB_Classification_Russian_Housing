# XGB_Classification_Russian_Housing

import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.base import TransformerMixin
import math
from sklearn import datasets, linear_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

%matplotlib inline

path = 'C:/Users/RAJNEESH TIWARI/Downloads/Russian Housing prediction'
train = pd.read_csv(os.path.join(path,"train.csv"))
test = pd.read_csv(os.path.join(path,"test.csv"))
macro = pd.read_csv(os.path.join(path,"macro.csv"))
submission = pd.read_csv(os.path.join(path,"sample_submission.csv"))

#train = pd.merge(train, macro, how='left', on='timestamp')
#test = pd.merge(test, macro, how='left', on='timestamp')
print(train.shape, test.shape)

trainsub = train[train.timestamp < '2015-01-01']
trainsub = trainsub[trainsub.product_type=="Investment"]

ind_1m = trainsub[trainsub.price_doc <= 1000000].index
ind_2m = trainsub[trainsub.price_doc == 2000000].index
ind_3m = trainsub[trainsub.price_doc == 3000000].index

train_index = set(train.index.copy())

for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
    ind_set = set(ind)
    ind_set_cut = ind.difference(set(ind[::gap]))

    train_index = train_index.difference(ind_set_cut)

train = train.loc[train_index]

test.dropna(how="all", axis=1)
train.dropna(how="all", axis=1)

col_test = test.columns
col_train = train.columns
col_train_unique = set(col_train)
intersection = [val for val in col_test if val in col_train_unique]
print (intersection)

price_doc = train['price_doc']
train = train[intersection]
test = test[intersection]
train['price_doc'] = price_doc

#price_bins = 3 ### 3 equal bins
price_labels = ['Low','Medium','High']
train['price_cat'] = pd.qcut(train['price_doc'], 3, retbins=False, labels=price_labels)

price_doc_V1 = train['price_doc']
train.drop('price_doc',axis=1,inplace=True)
train['price_cat'] = train['price_cat'].astype(object)
price_cateogorical = train['price_cat']

train.drop('price_cat',axis=1,inplace=True) 

combined_set = pd.concat([train,test],axis=0)
for feature in combined_set.columns: # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes # Replace strings with an integer

final_train = combined_set[:train.shape[0]] # Up to the last initial training set row
final_test = combined_set[train.shape[0]:] # Past the last initial training set row

y_train = price_cateogorical
y_train = pd.Categorical(y_train).codes

test_size = 0.33
X_tr, X_test, Y_tr, Y_test = train_test_split(final_train, y_train, test_size=test_size, random_state=1234)

xg_train = xgb.DMatrix(X_tr, label=Y_tr)
xg_test = xgb.DMatrix(X_test, label=Y_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification -- we will use softprob which is sigmoid kind of
param['objective'] = 'multi:softprob'
param["subsample"] = 0.8
param["colsample_bytree"] = 0.8
#param['eval_metric'] = ['error', 'logloss']
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3
param['eval_metric'] = ['mlogloss','merror']

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=30)
# get prediction
pred = bst.predict( xg_test )
