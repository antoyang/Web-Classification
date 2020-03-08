# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:30:48 2020

@author: Antoine
"""
import fasttext
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle

# Read keywords
keywords_list = open("test_keywords.pkl", "rb")
keywords_list = pickle.load(keywords_list)

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
ft = fasttext.load_model('cc.fr.300.bin')

# Get Embeddings
filenames = test_hosts
doc_embeddings = dict.fromkeys(test_hosts)
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

# Save Embeddings
with open('test_keywords_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)