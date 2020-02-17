import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GNN
from graph_based_approach.utils import accuracy, normalize_adjacency
import pickle
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

# Hyperparameters
epochs = 100
n_hidden_1 = 8
n_hidden_2 = 16
learning_rate = 0.01
dropout_rate = 0.2
n_class = 8

adjacency_matrix_filename = './norm_adj.gz'

####################################

# Data Preprocessing

classes = {"business/finance": 0, "education/research": 1, "entertainment": 2, "health/medical": 3, "news/press": 4,
           "politics/government/law": 5, "sports": 6, "tech/science": 7}

# Read training data
with open("../prototyping_data/reduced_train.csv", 'r') as f:
    train_data = f.read().splitlines()

x_train = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    x_train.append(int(host))
    y_train.append(classes[label.lower()])

# Read test data
# with open("../../data/test.csv", 'r') as f:
#    test_hosts = f.read().splitlines()

# Loads the karate network
G = nx.read_weighted_edgelist('../prototyping_data/reduced_edgelist.txt', delimiter=' ', nodetype=int,
                              create_using=nx.DiGraph())
print(G.number_of_nodes())
print(G.number_of_edges())

n = G.number_of_nodes()

print("ADJ BEG")
adj = nx.to_numpy_matrix(G)  # Obtains the adjacency matrix
print("ADJ MID")
adj = normalize_adjacency(adj)  # Normalizes the adjacency matrix
print("ADJ END")

####################################

# Features
#features = np.ones((n, n))

# Set the feature of all nodes to the same value
features = np.eye(n)  # Generates node features

# Yields indices to split data into training and test sets
idx = np.random.RandomState(seed=42).permutation(n)

# Transforms the numpy matrices/vectors to torch tensors
features = torch.FloatTensor(features)
features = features.to(device)
y_train = torch.LongTensor(y_train)
y_train = y_train.to(device)
adj = torch.FloatTensor(adj)
adj = adj.to(device)

# Creates the model and specifies the optimizer
model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_test_cut = len(x_train)//10 * 7

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    output = output[0]
    loss_train = F.nll_loss(output[:train_test_cut], y_train[:train_test_cut])
    acc_train = accuracy(output[:train_test_cut], y_train[:train_test_cut])
    # loss_train = F.nll_loss(output[x_train], y_train)
    # acc_train = accuracy(output[x_train], y_train)
    loss_train.backward()
    optimizer.step()

    model.eval()
    loss_test = F.nll_loss(output[train_test_cut:], y_train[train_test_cut:])
    acc_test = accuracy(output[train_test_cut:], y_train[train_test_cut:])
    print('Epoch: {:03d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))


t_total = time.time()
print("Start Train")
for epoch in range(epochs):
    train(epoch)

with open('graph_nn_model.pkl', 'wb') as modelfile:
    torch.save(model, modelfile)

model.eval()
output = model(features, adj)[0]
with open('results.pkl', 'wb') as resultfile:
    pickle.dump(output, resultfile)

classes_per_cluster = dict()
output = torch.argmax(output, 1)
output = output.numpy()
for cluster, true_class in zip(output, y_train):
    to_add = np.array([0 for _ in range(len(classes))])
    to_add[true_class] += 1
    classes_per_cluster[cluster] = classes_per_cluster.get(cluster, np.array([0 for _ in range(len(classes))])) + to_add


classes_per_cluster_test = dict()
for cluster, true_class in zip(output[train_test_cut:], y_train[train_test_cut:]):
    to_add = np.array([0 for _ in range(len(classes))])
    to_add[true_class] += 1
    classes_per_cluster_test[cluster] = classes_per_cluster_test.get(cluster, np.array([0 for _ in range(len(classes))])) + to_add

print(classes_per_cluster)
print(classes_per_cluster_test)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print()
