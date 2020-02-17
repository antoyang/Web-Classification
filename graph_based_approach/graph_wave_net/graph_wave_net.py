# Need to install the graphwave library in the same directory.


import networkx as nx
import numpy as np
import pickle

import matplotlib.pyplot as plt
from graph_based_approach.utils import draw_conf_mat
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from graphwave.graphwave import graphwave_alg
from sklearn.metrics import confusion_matrix, classification_report, log_loss




classes = {"business/finance": 0, "education/research": 1, "entertainment": 2, "health/medical": 3, "news/press": 4,
           "politics/government/law": 5, "sports": 6, "tech/science": 7}

color_map = ["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white"]


def get_embedding_prototype():
    with open('../prototyping_data/reduced_edgelist.txt') as edgelist:
        G = nx.read_weighted_edgelist(edgelist, delimiter=' ', nodetype=int, create_using=nx.DiGraph())

    chi, heat_print, taus = graphwave_alg(G, np.linspace(0, 10000, 10000), taus='auto', verbose=True)
    print(chi.shape)
    print(heat_print)

    pca = PCA(n_components=15)
    pca_data = pca.fit_transform(StandardScaler().fit_transform(chi))
    print("PCA variance explained:", pca.explained_variance_ratio_.sum())

    tsne = TSNE(n_components=2, perplexity=10)
    my_tsne_fit = tsne.fit_transform(pca_data)

    labels_dict = {}
    labels_list = []
    with open('../prototyping_data/reduced_train.csv') as labelslist:
        for line in labelslist:
            node_idx, label = line.replace("\n", "").split(',')
            labels_dict[int(node_idx)] = label
            labels_list.append(label)

    colors = []
    nodes = np.array(G.nodes)
    for idx in range(len(my_tsne_fit)):
        colors.append(color_map[classes.get(labels_dict.get(nodes[idx], "NA"))])

    fig, ax = plt.subplots()
    ax.scatter(my_tsne_fit[:, 0], my_tsne_fit[:, 1], s=5, c=colors)

    fig.suptitle('t-SNE visualization of node embeddings', fontsize=20)
    fig.set_size_inches(11, 7)
    fig.savefig('graphwave_node_embeddings.pdf', dpi=300)
    fig.show()

    with open("chi_file_proto.pkl", "wb") as chi_dump:
        pickle.dump(chi, chi_dump)
    with open("heat_print_file_proto.pkl", "wb") as heat_dump:
        pickle.dump(heat_print, heat_dump)
    with open("pca_embedding_proto.pkl", "wb") as pca_dump:
        pickle.dump(pca_data, pca_dump)
    with open("labels_proto.pkl", "wb") as label_dump:
        pickle.dump(labels_list, label_dump)

def get_embedding():
    with open('../../data/edgelist.txt') as edgelist:
        G = nx.read_weighted_edgelist(edgelist, delimiter=' ', nodetype=int, create_using=nx.DiGraph())

    chi, heat_print, taus = graphwave_alg(G, np.linspace(0, 10000, 10000), taus='auto', verbose=True)
    print(chi.shape)
    print(heat_print)

    pca = PCA(n_components=15)
    pca_data = pca.fit_transform(StandardScaler().fit_transform(chi))
    print("PCA variance explained:", pca.explained_variance_ratio_.sum())

    with open("chi_file.pkl", "wb") as chi_dump:
        pickle.dump(chi, chi_dump)
    with open("heat_print_file.pkl", "wb") as heat_dump:
        pickle.dump(heat_print, heat_dump)
    with open("pca_embedding.pkl", "wb") as pca_dump:
        pickle.dump(pca_data, pca_dump)


def load_embedding(proto=False):
    chi_filename = "chi_file.pkl"
    heat_filename = "heat_print_file.pkl"
    pca_filename = "pca_embedding.pkl"
    labels_filename = "labels.pkl"
    if proto:
        chi_filename = "chi_file_proto.pkl"
        heat_filename = "heat_print_file_proto.pkl"
        pca_filename = "pca_embedding_proto.pkl"
        labels_filename = "labels_proto.pkl"

    with open(chi_filename, "rb") as chi_dump:
        chi = pickle.load(chi_dump)
    with open(heat_filename, "rb") as heat_dump:
        heat_print = pickle.load(heat_dump)
    with open(pca_filename, "rb") as pca_dump:
        pca_data = pickle.load(pca_dump)
    with open(labels_filename, "rb") as label_dump:
        labels_list = pickle.load(label_dump)

    return chi, heat_print, pca_data, labels_list


def main():
    #get_embedding_prototype() # Should run this the first time then you can comment it out
    get_embedding()

    chi, heat_print, pca_data, labels_list = load_embedding(proto=True)
    valid_sample_per_class = 5
    valid_data = []
    valid_labels = []
    for i in range(len(labels_list)):
        label = labels_list[i]
        if valid_labels.count(label) < valid_sample_per_class:
            valid_labels.append(labels_list[i])
            np.delete(labels_list, i)

            valid_data.append(pca_data[i])
            np.delete(pca_data, i)

    # clf = LogisticRegression(solver='lbfgs', multi_class='auto', class_weight='balanced', max_iter=40000)
    clf = RandomForestClassifier(min_samples_split=3, min_samples_leaf=2, max_depth=15)
    clf.fit(pca_data, labels_list)
    y_pred_train = clf.predict(pca_data)
    y_pred_valid = clf.predict(valid_data)

    y_pred_train_prob = clf.predict_proba(pca_data)
    y_pred_valid_prob = clf.predict_proba(valid_data)

    print(classification_report(labels_list, y_pred_train))
    print("Training LogLoss:", log_loss(labels_list, y_pred_train_prob))

    print(classification_report(valid_labels, y_pred_valid))
    print("Valid LogLoss:", log_loss(valid_labels, y_pred_valid_prob))

    draw_conf_mat(confusion_matrix(labels_list, y_pred_train), classes.keys(), save=True,
                  file_name="conf_mat_graph_wave")
    draw_conf_mat(confusion_matrix(valid_labels, y_pred_valid), classes.keys(), save=True,
                  file_name="conf_mat_graph_wave_valid")


if __name__ == "__main__":
    main()
