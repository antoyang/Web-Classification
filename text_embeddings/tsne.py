# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:12:02 2020

@author: Antoine
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

colors = ['blue','red', 'green', 'yellow', 'brown', 'grey', 'black', 'magenta']

def visualize_doc_embeddings(my_doc_embs,my_colors,my_labels,my_name):
    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2,perplexity=10) #https://lvdmaaten.github.io/tsne/
    doc_embs_pca = my_pca.fit_transform(my_doc_embs) 
    doc_embs_tsne = my_tsne.fit_transform(doc_embs_pca)
    
    fig, ax = plt.subplots()
    
    for i, label in enumerate(list(set(my_labels))):
        idxs = [idx for idx,elt in enumerate(my_labels) if elt==label]
        ax.scatter(doc_embs_tsne[idxs,0], 
                   doc_embs_tsne[idxs,1], 
                   c = my_colors[i],
                   label=str(label),
                   alpha=0.7,
                   s=40)
    
    ax.legend(scatterpoints=1)
    fig.suptitle('t-SNE visualization doc embeddings',fontsize=15)
    fig.set_size_inches(11,7)
    fig.savefig(my_name + '.pdf')
    
# BOURRIN FASTEXT EMBEDDINGS

file = open('fasttext_bourrin_embeddings.pkl','rb')
embeddings = pickle.load(file)

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
embeddings_hosts = list()
count_nan = 0
for i, row in enumerate(train_data):
    host, label = row.split(",")
    if np.isnan(embeddings[host]).any():
        count_nan +=1
        continue
    """if embeddings[host] is None:
        count_nan +=1
        continue"""
    train_hosts.append(host)
    y_train.append(label.lower())
    embeddings_hosts.append(embeddings[host])
    # embeddings_hosts.append(embeddings[i].values)
    
visualize_doc_embeddings(embeddings_hosts,colors,y_train,"Bourrin_fasttext_Embeddings")

# KEYWORDS FASTEXT EMBEDDINGS

file = open('doc_keywords_embeddings.pkl','rb')
embeddings = pickle.load(file)

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
embeddings_hosts = list()
count_nan = 0
for i, row in enumerate(train_data):
    host, label = row.split(",")
    if embeddings[host] is None:
        count_nan +=1
        continue
    train_hosts.append(host)
    y_train.append(label.lower())
    embeddings_hosts.append(embeddings[host])

visualize_doc_embeddings(embeddings_hosts,colors,y_train,"Keywords_fasttext_Embeddings")

# TF-IDF EMBEDDINGS

embeddings = pd.read_csv('tfidf_emb_train.csv', header=None, dtype={0: str}).transpose()

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
embeddings_hosts = list()
count_nan = 0
for i, row in enumerate(train_data):
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())
    embeddings_hosts.append(embeddings[i].values)
    
visualize_doc_embeddings(embeddings_hosts,colors,y_train,"TFIDF_Embeddings")
