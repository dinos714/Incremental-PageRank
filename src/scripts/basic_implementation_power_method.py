import time
import argparse
import numpy as np
import scipy.sparse as sp


def load_graph(data_file_path):
    edges = []
    n_nodes = 0
    with open(data_file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                line = line.strip().split('\t')
                edges.append((int(line[0]), int(line[1])))
            elif line.startswith('# Nodes:'):
                n_nodes = int(line.split(' ')[2].strip())
    edges = np.array(edges)
    return edges, n_nodes


def build_adjacency_matrix(edges, n_nodes, p):
    G = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        G[j-1, i-1] = 1
    D = np.linalg.inv(np.diag(G.sum(axis=0)))
    e = np.ones(n_nodes)
    f = np.zeros(n_nodes)
    for i in range(n_nodes):
        if G[:, i].sum() != 0:
            f[i] = (1-p)/n_nodes
        else:
            f[i] = 1/n_nodes
    A = p * G @ D + e * f
    return A


def power_method(A, n_nodes, max_iter=100, tol=1e-6):
    x = np.random.rand(n_nodes)
    x /= np.linalg.norm(x, 1)
    for _ in range(max_iter):
        x_new = A @ x
        x_new /= np.linalg.norm(x_new, 1)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x


def main(data_file_path, show_top_k=5, p=0.85, max_iter=100, tol=1e-6):
    edges, n_nodes = load_graph(data_file_path)
    print(f"Loaded graph with {n_nodes} nodes and {len(edges)} edges from {data_file_path}.")
    print("\n=== Basic Implementation of Power Method PageRank ===")
    start_time = time.time()
    A = build_adjacency_matrix(edges, n_nodes, p)
    print("Adjacency matrix A built:\n", A)
    pagerank_vector = power_method(A, n_nodes, max_iter, tol)
    end_time = time.time()
    t = end_time - start_time
    print(f"\n=== PageRank vector got. Computed in {t:.4f} seconds ===\n")
    print(f"=== Top {show_top_k} nodes by PageRank ===")
    top_indices = np.argsort(pagerank_vector)[::-1][:show_top_k]
    for idx in top_indices:
        print(f"   Node {idx+1}: PageRank = {pagerank_vector[idx]:.4f}")
    print("\n=== Complete PageRank vector ===\n", pagerank_vector)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Basic Power Method PageRank Implementation')
    argparser.add_argument('--data_file_path', type=str, default='./data/wiki-Vote.txt', help='Path to the data file')
    argparser.add_argument('--show_top_k', type=int, default=5, help='Number of top nodes to display')
    argparser.add_argument('--p', type=float, default=0.85, help='Damping factor')
    args = argparser.parse_args()
    
    main(args.data_file_path, show_top_k=args.show_top_k, p=args.p)