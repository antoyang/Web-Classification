# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:57:55 2020

@author: flo-r
"""

import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from pprint import pprint

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
X = pd.read_csv("kpca_emb_train.csv", header=None)
Y = pd.read_csv("kpca_emb_test.csv", header=None)
print("train dimension:", X.shape)
print("test dimension:", Y.shape)

# Grid search for Random Forest, with 4-fold CV
penalty = ['l1', 'l2', 'elasticnet', 'none']
C = [.0001, .001, .01, .1, 1, 10, 100, 1000]
max_iter=[10000]
grid = {'penalty': penalty,
        'C': C,
        'max_iter': max_iter}
pprint(grid)

clf = LogisticRegression()
clf_grid = GridSearchCV(estimator=clf, param_grid=grid, scoring='neg_log_loss',
                        cv = 5, verbose=2, n_jobs = 12)
clf_grid.fit(X, y_train)
print('best CV score:', clf_grid.best_score_)
pprint(clf_grid.best_params_)

# Make predictions
y_pred = clf_grid.best_estimator_.predict_proba(Y)
with open('lr_graph_kpca.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf_grid.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)