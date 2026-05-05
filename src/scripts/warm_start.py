import time
import argparse
import numpy as np
import scipy.sparse as sp
from static_textook_implementation_power_method import load_graph, build_components, power_method_standard as power_method


def power_method_with_init_vec(G, D, e, f, n_nodes, p, x_init=None, max_iter=500, tol=1e-9):
    if x_init is None:
        x = np.random.rand(n_nodes)
    else:
        x = x_init.copy()        
    x /= np.linalg.norm(x, 1)
    iters = 0
    for _ in range(max_iter):
        iters += 1
        x_new = p * G @ (D * x) + e * (f @ x)
        x_new /= np.linalg.norm(x_new, 1)        
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x, iters


def perturb_graph(edges, n_nodes, add_ratio=0.001, drop_ratio=0.001):
    """
        Perturb the graph by randomly adding and dropping edges.
        - drop_ratio: the fraction of edges to drop
        - add_ratio: the fraction of new edges to add (relative to original edge count)
    """
    n_edges = len(edges)
    drop_count = int(n_edges * drop_ratio)
    keep_indices = np.random.choice(n_edges, n_edges - drop_count, replace=False)
    edges_kept = edges[keep_indices]
    add_count = int(n_edges * add_ratio)
    new_edges = np.random.randint(0, n_nodes, size=(add_count, 2))
    edges_new = np.vstack([edges_kept, new_edges])
    
    return edges_new, add_count, drop_count


def main(data_file_path, show_top_k, p, tol):
    print("="*60)
    print(f"{'WARM-START INCREMENTAL PAGERANK REPORT':^60}")
    print("="*60)
    print("[*] Loading original graph and calcutaing PageRank vector...")
    edges, n_nodes = load_graph(data_file_path)
    G, D, e, f, v = build_components(edges, n_nodes, p)
    x_old = power_method(G, D, e, f, n_nodes, p, max_iter=500, tol=tol)
    
    print("[*] Perturbating (adding and droping edges)...")
    edges_new, _, _ = perturb_graph(edges, n_nodes, add_ratio=0.005, drop_ratio=0.005)
    G_new, D_new, e_new, f_new, _ = build_components(edges_new, n_nodes, p)
    print("-" * 60)
    
    results = []

    # Method 1: Cold Start
    print("[1] Cold Start (Initializing with a random vector)...")
    start_time = time.perf_counter()
    x_cold, iters_cold = power_method_with_init_vec(G_new, D_new, e_new, f_new, n_nodes, p, x_init=None, tol=tol)
    t_cold = time.perf_counter() - start_time
    print(f"    Done! Iterations: {iters_cold}, Time: {t_cold:.6f}s")
    results.append(("Cold Start", t_cold, iters_cold, x_cold))
    
    # Method 2: Warm Start
    print("\n[2] Warm-Start (Initialize with the previous PageRank vector)...")
    start_time = time.perf_counter()
    x_warm, iters_warm = power_method_with_init_vec(
        G_new, D_new, e_new, f_new, n_nodes, p, x_init=x_old, tol=tol
    )
    t_warm = time.perf_counter() - start_time
    print(f"    Done! Iterations: {iters_warm}, Time: {t_warm:.6f}s")
    results.append(("Warm-Start", t_warm, iters_warm, x_warm))

    # --- Comparisons and Ranking ---
    for name, duration, iters, vec in results:
        print("\n" + "#"*60)
        print(f"  METHOD: {name}")
        print(f"  Execution Time: {duration:.6f} seconds")
        print(f"  Iterations    : {iters}")
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

    # Consistency Check
    print("="*60)
    print("[3] Consistency & Performance Check:")
    diff = np.linalg.norm(x_cold - x_warm, 1)
    print(f"    ||x_cold - x_warm||_1 = {diff:.4e} (should be close to 0)")
    print("-" * 60)
    print(f"Conclusion:")
    print(f" -> Iteration number reduced: {iters_cold - iters_warm} ({(1 - iters_warm/iters_cold)*100:.2f}%)")
    print(f" -> Running time reduced    : {t_cold - t_warm:.6f}s ({(1 - t_warm/t_cold)*100:.2f}%)")
    print("="*60)
    print(f"{'END OF REPORT':^60}")
    print("="*60)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Warm Start vs Cold Start PageRank')
    argparser.add_argument('--data_file_path', type=str, default='./data/Wiki-Vote.txt', help='Path to the data file')
    argparser.add_argument('--show_top_k', type=int, default=5, help='Number of top nodes to display')
    argparser.add_argument('--p', type=float, default=0.85, help='Damping factor')
    argparser.add_argument('--tol', type=float, default=1e-9, help='Convergence tolerance')
    args = argparser.parse_args()
    
    main(args.data_file_path, args.show_top_k, args.p, args.tol)