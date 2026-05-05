import time
import argparse
import numpy as np
import scipy.sparse as sp


def load_graph(data_file_path):
    edges = []
    node_map = {}
    node_counter = 0
    with open(data_file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split('\t')
                u, v = int(parts[0]), int(parts[1])
                if u not in node_map:
                    node_map[u] = node_counter
                    node_counter += 1
                if v not in node_map:
                    node_map[v] = node_counter
                    node_counter += 1
                edges.append((node_map[u], node_map[v]))
    edges = np.array(edges)
    n_nodes = node_counter
    return edges, n_nodes


def build_adjacency_matrix(edges, n_nodes, p):
    G = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        G[j, i] = 1
    out_degrees = G.sum(axis=0)
    D_diag = np.ones(n_nodes)
    non_zero_mask = out_degrees > 0
    D_diag[non_zero_mask] = 1.0 / out_degrees[non_zero_mask]
    D = np.diag(D_diag)
    e = np.ones(n_nodes)
    f = np.full(n_nodes, 1.0 / n_nodes)
    f[non_zero_mask] = (1 - p) / n_nodes
    A = p * G @ D + np.outer(e, f)
    return A


def build_components(edges, n_nodes, p):
    cols = edges[:, 0]
    rows = edges[:, 1]
    data = np.ones(len(edges))
    G = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    out_degrees = np.array(G.sum(axis=0)).flatten()
    D = np.ones(n_nodes)
    non_zero_mask = out_degrees > 0
    D[non_zero_mask] = 1.0 / out_degrees[non_zero_mask]
    e = np.ones(n_nodes)
    f = np.zeros(n_nodes)
    for i in range(n_nodes):
        if out_degrees[i] != 0:
            f[i] = (1-p)/n_nodes
        else:
            f[i] = 1/n_nodes
    v = e * 1/n_nodes
    return G, D, e, f, v


def power_method_adjacency_matrix_A(A, n_nodes, max_iter=100, tol=1e-6):
    x = np.random.rand(n_nodes)
    x /= np.linalg.norm(x, 1)
    for _ in range(max_iter):
        x_new = A @ x
        x_new /= np.linalg.norm(x_new, 1)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x


def power_method_standard(G, D, e, f, n_nodes, p, max_iter=100, tol=1e-6):
    x = np.random.rand(n_nodes)
    x /= np.linalg.norm(x, 1)
    for _ in range(max_iter):
        x_new = p * G @ (D * x) + e * (f @ x)
        x_new /= np.linalg.norm(x_new, 1)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x


'''
    Use this when there is no dangling node.
    A = pM + (1−p)evT
        G = graph adjacency matrix
        D = diagonal out-degree matrix
        e = vector of ones
        M = GD
        v = e/n
    x = pMx + (1-p)evTx
'''
def power_method_dangling_free(G, D, e, v, n_nodes, p, max_iter=100, tol=1e-6):
    x = np.random.rand(n_nodes)
    x /= np.linalg.norm(x, 1)
    for _ in range(max_iter):
        x_new = p * G @ (D * x) + (1-p) * e * (v @ x)
        x_new /= np.linalg.norm(x_new, 1)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x


def main(data_file_path, show_top_k=5, p=0.85, max_iter=100, tol=1e-6):
    edges, n_nodes = load_graph(data_file_path)
    print("="*60)
    print(f"{'PAGERANK COMPUTATION REPORT':^60}")
    print("="*60)
    print(f"[*] Data Source  : {data_file_path}")
    print(f"[*] Nodes        : {n_nodes}")
    print(f"[*] Edges        : {len(edges)}")
    print(f"[*] Parameters   : p={p}, tol={tol}")
    print("-"*60)

    results = []

    # --- 1. Basic Implementation ---
    print(f"\n[1] Running: Basic Implementation (Dense Matrix)...")
    start_time = time.time()
    A = build_adjacency_matrix(edges, n_nodes, p)
    pagerank_vector = power_method_adjacency_matrix_A(A, n_nodes, max_iter, tol)
    t1 = time.time() - start_time
    results.append(("Basic (Dense)", t1, pagerank_vector))
    print(f"    Done! Time: {t1:.4f}s")

    # --- 2. Component-wise Implementation ---
    print(f"\n[2] Running: Component-wise Implementation (CSR Sparse)...")
    start_time = time.time()
    G, D, e, f, v = build_components(edges, n_nodes, p)
    pagerank_vector_comp = power_method_standard(G, D, e, f, n_nodes, p, max_iter, tol)
    # pagerank_vector_comp = power_method_standard(G, D, e, v, n_nodes, p, max_iter, tol)
    t2 = time.time() - start_time
    results.append(("Component-wise (CSR)", t2, pagerank_vector_comp))
    print(f"    Done! Time: {t2:.4f}s")

    # --- Comparisons and Ranking ---
    for name, duration, vec in results:
        print("\n" + "#"*60)
        print(f"  METHOD: {name}")
        print(f"  Execution Time: {duration:.4f} seconds")
        print("#"*60)
        print(f"\n  Top {show_top_k} Ranking:")
        print(f"  {'-'*35}")
        print(f"  {'Rank':<8} | {'Node ID':<10} | {'Score':<10}")
        print(f"  {'-'*35}")
        top_indices = np.argsort(vec)[::-1][:show_top_k]
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank:<8} | {idx+1:<10} | {vec[idx]:.6f}")
        print(f"  {'-'*35}")
        if len(vec) > 10:
            formatted_vec = "[" + ", ".join([f"{v:.4f}" for v in vec[:4]]) + " ... " + ", ".join([f"{v:.4f}" for v in vec[-4:]]) + "]"
        else:
            formatted_vec = vec
        print(f"\n  Vector Preview:\n  {formatted_vec}\n")

    print("="*60)
    print(f"{'END OF REPORT':^60}")
    print("="*60)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Basic Power Method PageRank Implementation')
    argparser.add_argument('--data_file_path', type=str, default='./data/Wiki-Vote.txt', help='Path to the data file')
    argparser.add_argument('--show_top_k', type=int, default=5, help='Number of top nodes to display')
    argparser.add_argument('--p', type=float, default=0.85, help='Damping factor')
    args = argparser.parse_args()
    
    main(args.data_file_path, show_top_k=args.show_top_k, p=args.p)