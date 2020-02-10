# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:30:48 2020

@author: Antoine
"""

from gensim.models import word2vec
import fasttext
import fasttext.util
import os
import string
import operator
import nltk
from nltk.corpus import stopwords
from library import clean_text_simple,terms_to_graph
import numpy as np
import pickle

keywords_list = open("keywords.pkl", "rb")
keywords_list = pickle.load(keywords_list)
token_list = open('token_list.pkl','rb')
token_list = pickle.load(token_list)

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
    
# Tokenize web site texts and embed documents
stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('french')
punct = string.punctuation.replace('-', '')

# Load fast text
# ft = fasttext.load_model('cc.fr.300.bin')

filenames = train_hosts
doc_embeddings = dict.fromkeys(train_hosts)
for i,file in enumerate(filenames):
    keywords = keywords_list[i]
    if len(keywords)<1:
        continue
    doc_embedding = np.zeros(300)
    for token in keywords:
        doc_embedding += ft.get_word_vector(token)
    doc_embedding /= len(keywords)
    doc_embeddings[file] = doc_embedding
    if i%100==0:
        print(i)
    
with open('doc_keywords_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)