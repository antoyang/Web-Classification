import argparse
import numpy as np

from tqdm import tqdm
from scipy.linalg import svd
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
from graph_based_approach.utils import read_edgelist, extract_egonets, write_to_file

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-edgelist', default='../prototyping_data/reduced_edgelist.txt',
                    help='Path to the edgelist.')
parser.add_argument('--delimiter', default=' ',
                    help='The string used to separate values.')
parser.add_argument('--path-to-output-file', default='reduced_embedding2.txt',
                    help='Path to output file')
parser.add_argument('--radius', type=int, default=2,
                    help='Maximum radius of ego-networks.')
parser.add_argument('--dim', type=int, default=10,
                    help='Dimensionality of the embeddings.')
parser.add_argument('--kernel', default='shortest_path',
                    help='Graph kernel (shortest_path or weisfeiler_lehman).')


def segk(nodes, edgelist, radius, dim, kernel):
    n = len(nodes)

    if kernel == 'shortest_path':
        gk = [ShortestPath(normalize=True, with_labels=True) for i in range(radius)]
    elif kernel == 'weisfeiler_lehman':
        gk = [WeisfeilerLehman(n_iter=4, normalize=True, base_kernel=VertexHistogram) for i in range(radius)]
    else:
        raise ValueError('Use a valid kernel!!')

    idx = np.random.permutation(n)
    sampled_nodes = [nodes[idx[i]] for i in range(dim)]
    remaining_nodes = [nodes[idx[i]] for i in range(dim, len(nodes))]

    egonet_edges, egonet_node_labels = extract_egonets(edgelist, radius)

    E = np.zeros((n, dim))

    K = np.zeros((dim, dim))
    K_prev = np.ones((dim, dim))

    for i in range(1, radius + 1):
        Gs = list()
        with tqdm(total=len(sampled_nodes)) as pbar:
            for node in sampled_nodes:
                node_labels = {v: egonet_node_labels[node][v] for v in egonet_node_labels[node] if
                               egonet_node_labels[node][v] <= i}
                edges = list()
                for edge in egonet_edges[node]:
                    if edge[0] in node_labels and edge[1] in node_labels:
                        edges.append((edge[0], edge[1]))
                        edges.append((edge[1], edge[0]))
                Gs.append(Graph(edges, node_labels=node_labels))
                pbar.update(1)

        K_i = gk[i - 1].fit_transform(Gs)
        K_i = np.multiply(K_prev, K_i)
        K += K_i
        K_prev = K_i

    U, S, V = svd(K)
    S = np.maximum(S, 1e-12)
    Norm = np.dot(U * 1. / np.sqrt(S), V)
    E[idx[:dim], :] = np.dot(K, Norm.T)

    K = np.zeros((n - dim, dim))
    K_prev = np.ones((n - dim, dim))

    for i in range(1, radius + 1):
        Gs = list()
        with tqdm(total=len(sampled_nodes)) as pbar:
            for node in remaining_nodes:
                node_labels = {v: egonet_node_labels[node][v] for v in egonet_node_labels[node] if
                               egonet_node_labels[node][v] <= i}
                edges = list()
                for edge in egonet_edges[node]:
                    if edge[0] in node_labels and edge[1] in node_labels:
                        edges.append((edge[0], edge[1]))
                        edges.append((edge[1], edge[0]))
                Gs.append(Graph(edges, node_labels=node_labels))
                pbar.update(1)

        K_i = gk[i - 1].transform(Gs)
        K_i = np.multiply(K_prev, K_i)
        K += K_i
        K_prev = K_i

    E[idx[dim:], :] = np.dot(K, Norm.T)

    return E


def main():
    args = parser.parse_args()

    nodes, edgelist = read_edgelist(args.path_to_edgelist, args.delimiter, cols=3)
    embeddings = segk(nodes, edgelist, args.radius, args.dim, args.kernel)
    write_to_file(args.path_to_output_file, nodes, embeddings)


if __name__ == "__main__":
    main()
