# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:57:55 2020

@author: flo-r
"""

import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import os
import pandas as pd
from pprint import pprint

os.chdir('C:/Users/flo-r/Desktop/Cours MVA/S1/ALTEGRAD/competition/Web-Classification')

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()
train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())
    
# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()

# Proportion of each class
for l in set(y_train):
    print("proportion of {}: {:.2f}%".format(l, 100*np.mean([l == w for w in y_train])))

# Read embeddings
X = pd.read_csv("tfidf_emb_train.csv", header=None)
Y = pd.read_csv("tfidf_emb_test.csv", header=None)
print("train dimension:", X.shape)
print("test dimension:", Y.shape)

# Grid search for Random Forest, with 4-fold CV
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 4)]
max_features = [.4, .5, .6, .7, .75, .8, .85, .9, .95]
max_depth = [int(x) for x in np.linspace(20, 100, num = 10)]
max_depth.append(None)
min_samples_leaf = [3, 8, 10, 15, 20, 50]
class_weight = ['balanced', 'balanced_subsample', None]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               'class_weight': class_weight}
pprint(random_grid)

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30,
                               scoring='neg_log_loss', cv = 3, verbose=2, n_jobs = 12)
rf_random.fit(X, y_train)
print('best CV score:', rf_random.best_score_)

pprint(rf_random.best_params_)

# Preds
y_pred = rf_random.best_estimator_.predictpredict_proba(Y)
with open('rf_tfidf_baseline.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = rf_random.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)