# Need to install the graphwave library in the same directory.


import networkx as nx
import numpy as np
import pickle
import csv

import matplotlib.pyplot as plt
from graph_based_approach.utils import draw_conf_mat
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from graphwave.graphwave import graphwave_alg
from sklearn.metrics import confusion_matrix, classification_report, log_loss

classes = {"business/finance": 0, "education/research": 1, "entertainment": 2, "health/medical": 3, "news/press": 4,
           "politics/government/law": 5, "sports": 6, "tech/science": 7}

color_map = ["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white"]


def get_embedding_reduced_train():
    needed_nodes = []
    labels_dict = {}
    labels_list = []
    with open('../prototyping_data/reduced_train2.csv') as nodelist:
        for line in nodelist:
            node_idx, label = line.replace("\n", "").split(',')
            needed_nodes.append(int(node_idx))
            labels_dict[int(node_idx)] = label
            labels_list.append(label)

    with open('../prototyping_data/reduced_train_edgelist.txt') as edgelist:
        G = nx.read_weighted_edgelist(edgelist, delimiter=' ', nodetype=int, create_using=nx.DiGraph())

    chi, heat_print, taus = graphwave_alg(G, np.linspace(0, 100, 1000), taus='auto', verbose=True)

    my_chi = []
    nodes = np.array(G.nodes)
    for i in range(len(nodes)):
        if nodes[i] in needed_nodes:
            my_chi.append(chi[i])

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(StandardScaler().fit_transform(my_chi))
    print("PCA variance explained:", pca.explained_variance_ratio_.sum())

    tsne = TSNE(n_components=2, perplexity=10)
    my_tsne_fit = tsne.fit_transform(pca_data)

    colors = []
    for idx in range(len(my_tsne_fit)):
        colors.append(color_map[classes.get(labels_dict.get(needed_nodes[idx], "NA"))])

    fig, ax = plt.subplots()
    ax.scatter(my_tsne_fit[:, 0], my_tsne_fit[:, 1], s=5, c=colors)
    ax.legend()
    fig.suptitle('t-SNE visualization of node embeddings', fontsize=20)
    fig.set_size_inches(11, 7)
    fig.savefig('graphwave_node_embeddings.pdf', dpi=300)
    fig.show()

    with open("pca_embedding_proto_train.pkl", "wb") as pca_dump:
        pickle.dump(pca_data, pca_dump)
    with open("labels_proto_train.pkl", "wb") as label_dump:
        pickle.dump(labels_list, label_dump)


def get_embedding_reduced_test():
    needed_nodes = []
    with open('../prototyping_data/reduced_test2.csv') as nodelist:
        for line in nodelist:
            node_idx = line.replace("\n", "")
            needed_nodes.append(int(node_idx))

    with open('../prototyping_data/reduced_test_edgelist.txt') as edgelist:
        G = nx.read_weighted_edgelist(edgelist, delimiter=' ', nodetype=int, create_using=nx.DiGraph())

    chi, heat_print, taus = graphwave_alg(G, np.linspace(0, 100, 1000), taus='auto', verbose=True)

    my_chi = []
    nodes = np.array(G.nodes)
    for i in range(len(nodes)):
        if nodes[i] in needed_nodes:
            my_chi.append(chi[i])

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(StandardScaler().fit_transform(my_chi))
    print("PCA variance explained:", pca.explained_variance_ratio_.sum())

    with open("pca_embedding_proto_test.pkl", "wb") as pca_dump:
        pickle.dump(pca_data, pca_dump)


def load_embedding(training_data=True):
    pca_filename = "pca_embedding_proto_test.pkl"

    if training_data:
        pca_filename = "pca_embedding_proto_train.pkl"

    with open(pca_filename, "rb") as pca_dump:
        pca_data = pickle.load(pca_dump)

    labels_list = []
    if training_data:
        labels_filename = "labels_proto_train.pkl"
        with open(labels_filename, "rb") as label_dump:
            labels_list = pickle.load(label_dump)

    return pca_data, labels_list


def main():
    #get_embedding_reduced_train()  # Should run this the first time then you can comment it out
    #get_embedding_reduced_test()

    pca_data, labels_list = load_embedding(training_data=True)
    pca_data_test, _ = load_embedding(training_data=False)

    valid_sample_per_class = 10
    valid_data = []
    valid_labels = []
    i = 0
    while i < len(labels_list):
        label = labels_list[i]
        if valid_labels.count(label) < valid_sample_per_class:
            valid_labels.append(labels_list[i])

            labels_list.pop(i)

            valid_data.append(pca_data[i])
            pca_data = np.delete(pca_data, i, axis=0)

            i -= 1
        i += 1

    #clf = LogisticRegression(solver='lbfgs', multi_class='auto', class_weight='balanced', max_iter=100000)
    clf = RandomForestClassifier(class_weight="balanced", min_impurity_decrease=0.0018, max_depth=7)
    #clf = MLPClassifier(hidden_layer_sizes=(30, 50, 50), learning_rate_init=0.0001,
    #                    validation_fraction=0.5, n_iter_no_change=200, solver='lbfgs')
    clf.fit(pca_data, labels_list)
    y_pred_train = clf.predict(pca_data)
    y_pred_valid = clf.predict(valid_data)

    y_pred_train_prob = clf.predict_proba(pca_data)
    y_pred_valid_prob = clf.predict_proba(valid_data)
    y_pred_test_prob = clf.predict_proba(pca_data_test)

    print(classification_report(labels_list, y_pred_train))
    print("Training LogLoss:", log_loss(labels_list, y_pred_train_prob))

    print(classification_report(valid_labels, y_pred_valid))
    print("Valid LogLoss:", log_loss(valid_labels, y_pred_valid_prob))

    draw_conf_mat(confusion_matrix(labels_list, y_pred_train), classes.keys(), save=True,
                  file_name="conf_mat_graph_wave_train")
    draw_conf_mat(confusion_matrix(valid_labels, y_pred_valid), classes.keys(), save=True,
                  file_name="conf_mat_graph_wave_valid")

    with open("result.pkl", "wb") as result_file:
        pickle.dump(y_pred_test_prob, result_file)

    needed_nodes = []
    with open('../prototyping_data/reduced_test2.csv') as nodelist:
        for line in nodelist:
            node_idx = line.replace("\n", "")
            needed_nodes.append(int(node_idx))

    # Write predictions to a file
    with open("graph_wave.csv", "w") as result_file:
        writer = csv.writer(result_file, delimiter=',')
        lst = clf.classes_.tolist()
        lst.insert(0, "Host")
        writer.writerow(lst)
        for idx, pred in enumerate(y_pred_test_prob):
            lst = y_pred_test_prob[idx, :].tolist()
            lst.insert(0, needed_nodes[idx])
            writer.writerow(lst)


if __name__ == "__main__":
    main()
