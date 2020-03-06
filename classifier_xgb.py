# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:57:55 2020

@author: flo-r
"""

import csv
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import os
import pandas as pd
from pprint import pprint

os.chdir('C:/Users/flo-r/Desktop/Cours MVA/S1/ALTEGRAD/competition/Web-Classification')

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()
label_train = list()
for row in train_data:
    _, label = row.split(",")
    label_train.append(label.lower())
    
# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()

# Read embeddings
train_emb = pd.read_csv("tfidf_emb_train.csv", header=None)
test_emb = pd.read_csv("tfidf_emb_test.csv", header=None)
print("train dimension:", train_emb.shape)
print("test dimension:", test_emb.shape)

# get valid set
X_train, X_val, y_train, y_val = train_test_split(train_emb,
                                                  label_train,
                                                  test_size=.1,
                                                  stratify=label_train)

# Random grid
n_estimators = [int(x) for x in np.linspace(100, 10000, 100)]
learning_rate = [.5, .2, .1, .07, .05, .04, .03, .02, .01]
gamma = [0, .01, .05, .1, .5, 1, 1.5, 2]
max_depth = [3, 4, 5, 6, 7, 8, 10, 13, 15, 20, 30, 40]
min_child_weight = [int(x) for x in np.arange(0, 11)]
subsample = [.7, .75, .8, .85, .9, .95, 1]
colsample_bytree = [.8, .85, .9, .95, 1]
colsample_bylevel = [.8, .85, .9, .95, 1]
colsample_bynode = [.8, .85, .9, .95, 1]
random_grid = {
    'n_estimators' : n_estimators,
    'learning_rate': learning_rate,
    'gamma': gamma,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'colsample_bylevel': colsample_bylevel,
    'colsample_bynode': colsample_bynode
    }
pprint(random_grid)

# Train on grid
xgb_clf = XGBClassifier()
xgb_random = RandomizedSearchCV(estimator=xgb_clf,
                                param_distributions=random_grid,
                                n_iter=100,
                                scoring='neg_log_loss',
                                cv=10, verbose=2, n_jobs = 12)
xgb_random.fit(X_train, y_train, early_stopping_rounds=10,
               eval_set = [(X_val, y_val)])
print('best CV score:', xgb_random.best_score_)
pprint(xgb_random.best_params_)

# Preds
y_pred = xgb_random.best_estimator_.predict_proba(test_emb)
with open('xgb_tfidf_baseline.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = xgb_random.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)