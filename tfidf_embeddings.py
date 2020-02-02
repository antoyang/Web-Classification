# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:53:43 2020

@author: flo-r
"""

import os
import csv
import string
import nltk
import codecs
from os import path
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LogisticRegression
from numpy import savetxt
from library import clean_text_simple


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

filenames = train_hosts + test_hosts
token_list = []
for (i, file) in enumerate(filenames):
    if i % 100 == 0:
        print('file {} over {}'.format(i, len(filenames)))
    with codecs.open(path.join('text/text/', file), encoding='latin-1') as f: 
        text = f.read().replace("\n", "")
    tokens = clean_text_simple(text, my_stopwords=stpwds, punct=punct, remove_stopwords=True, pos_filtering=True, stemming=True)
    token_list.append(tokens)

# Vocabulary length
raw = [elt for element in token_list for elt  in element]
print("Vocabulary of length", len(set(raw)))


# TF-IDF from tokenized text http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    encoding='latin-1',
    decode_error='ignore',
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None,
    min_df=10,
    max_df=1000,
    max_features=10000) 

tfidf_embeddings = tfidf.fit_transform(token_list)

print('50 first tokens', tfidf.get_feature_names()[:50])


# PCA
pca = SparsePCA(n_components=48)
pca_embeddings = pca.fit_transform(tfidf_embeddings.toarray())

savetxt('tfidf_emb_train.csv', pca_embeddings[:len(train_hosts)], delimiter=',')
savetxt('tfidf_emb_test.csv', pca_embeddings[len(train_hosts):], delimiter=',')


# Use logistic regression to classify the webpages of the test set
clf = LogisticRegression(solver='lbfgs', multi_class='auto')
clf.fit(pca_embeddings[:len(train_hosts)], y_train)

y_pred = clf.predict_proba(pca_embeddings[len(train_hosts):])

# Write predictions to a file
with open('tfidf_emb_baseline.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)