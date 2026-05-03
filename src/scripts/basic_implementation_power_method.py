import time
import argparse
import numpy as np
import scipy.sparse as sp


def load_graph(data_file_path):
    pass


def power_method(edges, n_nodes, p=0.85, max_iter=100, tol=1e-6):
    pass


def main(data_file_path, show_top_k=5, p=0.85):
    edges, n_nodes = load_graph(data_file_path)
    print(f"Loaded graph with {n_nodes} nodes and {len(edges)} edges from {data_file_path}.")
    print("\n=== Basic Power Method PageRank ===")
    start_time = time.time()
    pagerank_vector = power_method(edges, n_nodes, p)
    end_time = time.time()
    t = end_time - start_time
    print(f"Computed in {t:.4f} seconds")
    print(f"Top {show_top_k} nodes by PageRank:")
    top_indices = np.argsort(pagerank_vector)[::-1][:show_top_k]
    for idx in top_indices:
        print(f"Node {idx}: PageRank = {pagerank_vector[idx]:.6f}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Basic Power Method PageRank Implementation')
    argparser.add_argument('--data_file_path', type=str, default='../../data/wiki-Vote.txt', help='Path to the data file')
    argparser.add_argument('--show_top_k', type=int, default=5, help='Number of top nodes to display')
    argparser.add_argument('--p', type=float, default=0.85, help='Damping factor')
    args = argparser.parse_args()
    
    main(args.data_file_path, show_top_k=args.show_top_k, p=args.p)