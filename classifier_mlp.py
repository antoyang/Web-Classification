# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:05:05 2020

@author: flo-r
"""

import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
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

# Grid search mlp
grid = {'max_iter': [10000],
        'alpha': 10.0 ** -np.arange(1, 6),
        'hidden_layer_sizes': [(100,), (100, 100)],
        'learning_rate_init': [.005, .001, .0005],
        'early_stopping': [True],
        'validation_fraction': [.1, .2],
        'n_iter_no_change': [5, 10]}
pprint(grid)
mlp_grid = GridSearchCV(estimator=MLPClassifier(), param_grid=grid, scoring='neg_log_loss',
                        cv = 5, verbose=2, n_jobs = 10)
mlp_grid.fit(X, y_train)
print('best CV score:', mlp_grid.best_score_)
pprint(mlp_grid.best_params_)

# preds
y_pred = mlp_grid.best_estimator_.predict_proba(Y)
with open('mlp_tfidf_baseline.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = mlp_grid.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)