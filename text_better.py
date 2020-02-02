# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 00:29:51 2020

@author: Antoine
"""

import os
import string
import operator
import nltk
from nltk.corpus import stopwords
from library import clean_text_simple,terms_to_graph

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
    
# Tokenize web site texts
stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('french')
punct = string.punctuation.replace('-', '')

path = 'text/text'
# filenames = os.listdir(path)
filenames = train_hosts
token_list = []
for file in filenames:
    with open(path+'/'+file, 'r', encoding='latin-1') as f: 
        #if file == "10013":
            #import ipdb;ipdb.set_trace()
        text = f.read().splitlines()
        text = ' '.join(text)
    print(file)
    tokens = clean_text_simple(text, my_stopwords=stpwds, punct=punct, remove_stopwords=True, pos_filtering=True, stemming=True)
    tokens = list(set(tokens))
    token_list.append(tokens)

# PageRank keyword extraction
gs = [terms_to_graph(tokens, 4) for tokens in token_list]
nb_keywords = 20
keywords = []

for i, g in enumerate(gs):
    # PageRank
    print(i)
    pr_scores = zip(g.vs['name'],g.pagerank())
    pr_scores = sorted(pr_scores, key=operator.itemgetter(1), reverse=True) # in decreasing order
    numb_to_retain = 20 # retain top 'my_percentage' % words as keywords
    keywords.append([tuple[0] for tuple in pr_scores[:numb_to_retain]])
