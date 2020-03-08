# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:08:46 2020

@author: Antoine
"""
import fasttext
import fasttext.util
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle
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
    
# Tokenize web site texts and embed documents
stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('french')
punct = string.punctuation.replace('-', '')

# Load fasttext
ft = fasttext.load_model('cc.fr.300.bin')

# Get Embeddings
path = "text/text"
filenames = train_hosts
token_list = []
doc_embeddings = dict.fromkeys(train_hosts)
for file in filenames:
    with open(path+'/'+file, 'r', encoding='latin-1') as f: 
        text = f.read().splitlines()
        text = ' '.join(text)
    print(file)
    tokens = clean_text_simple(text, my_stopwords=stpwds, punct=punct, remove_stopwords=True, pos_filtering=True, stemming=True)
    tokens = list(set(tokens))
    token_list.append(tokens)
    doc_embedding = np.zeros(300)
    for token in tokens:
        doc_embedding += ft.get_word_vector(token)
    doc_embedding /= max(len(tokens),1)
    print(len(tokens))
    doc_embeddings[file] = doc_embedding
    
# Save Embeddings
with open('fasttext_bourrin_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)
    
# Same for test set
filenames = test_hosts
token_list = []
doc_embeddings = dict.fromkeys(test_hosts)
for file in filenames:
    with open(path+'/'+file, 'r', encoding='latin-1') as f: 
        text = f.read().splitlines()
        text = ' '.join(text)
    print(file)
    tokens = clean_text_simple(text, my_stopwords=stpwds, punct=punct, remove_stopwords=True, pos_filtering=True, stemming=True)
    tokens = list(set(tokens))
    token_list.append(tokens)
    doc_embedding = np.zeros(300)
    for token in tokens:
        doc_embedding += ft.get_word_vector(token)
    doc_embedding /= max(len(tokens),1)
    print(len(tokens))
    doc_embeddings[file] = doc_embedding
    
# Save Embeddings
with open('test_fasttext_bourrin_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)

    
    