import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from warm_start import perturb_graph, power_method_with_init_vec
from static_textook_implementation_power_method import load_graph, build_components


def vectorized_local_push(G_new, D_new, f_new, x_old, n_nodes, p, tol=1e-9):
    Ax = p * G_new @ (D_new * x_old) + np.dot(f_new, x_old)
    r = Ax - x_old
    x = x_old.copy()
    iters = 0
    while np.linalg.norm(r, 1) > tol:
        iters += 1
        x += r
        r = p * G_new @ (D_new * r) + np.dot(f_new, r)        
    x /= np.linalg.norm(x, 1)
    return x, iters


def main(args):
    print("="*60)
    print(f"{'INCREMENTAL PAGERANK SCALABILITY TEST':^60}")
    print("="*60)

    edges, n_nodes = load_graph(args.data_file_path)
    G, D, e, f, _ = build_components(edges, n_nodes, args.p)
    
    print("[*] 1/3 Calculating original converged PageRank vector...")
    x_old, _ = power_method_with_init_vec(G, D, e, f, n_nodes, args.p, tol=args.tol)
    
    print(f"[*] 2/3 Simulating network topology changes...")
    edges_new, added, dropped = perturb_graph(edges, n_nodes, args.add_ratio, args.drop_ratio)
    G_new, D_new, e_new, f_new, _ = build_components(edges_new, n_nodes, args.p)
    
    print(f"    -> Original Edges : {len(edges)}")
    print(f"    -> Added Edges    : {added} ({(args.add_ratio*100):.2f}%)")
    print(f"    -> Dropped Edges  : {dropped} ({(args.drop_ratio*100):.2f}%)")
    print(f"    -> New Total Edges: {len(edges_new)}")
    
    print("\n[*] 3/3 Running PageRank Updates...")
    
    results = []

    # Method: Cold Restart
    start_time = time.perf_counter()
    x_cold, iters_cold = power_method_with_init_vec(G_new, D_new, e_new, f_new, n_nodes, args.p, tol=args.tol)
    t_cold = time.perf_counter() - start_time
    print(f"\n[1] Cold Restart (Scratch)\n    -> Iters: {iters_cold:03d} | Time: {t_cold:.6f}s")
    results.append(("Cold Start", t_cold, iters_cold, x_cold))
    
    # Method: Warm-Start
    start_time = time.perf_counter()
    x_warm, iters_warm = power_method_with_init_vec(G_new, D_new, e_new, f_new, n_nodes, args.p, tol=args.tol, x_init=x_old)
    t_warm = time.perf_counter() - start_time
    print(f"\n[2] Warm-Start (Old x as Init)\n    -> Iters: {iters_warm:03d} | Time: {t_warm:.6f}s")
    results.append(("Warm-Start", t_warm, iters_warm, x_warm))
    
    # Method: local Push
    start_time = time.perf_counter()
    x_vec, iters_vec = vectorized_local_push(G_new, D_new, f_new, x_old, n_nodes, args.p, tol=args.tol)
    t_vec = time.perf_counter() - start_time
    print(f"\n[3] Local Push\n    -> Iters: {iters_vec:03d} | Time: {t_vec:.6f}s")
    results.append(("Local Push", t_vec, iters_vec, x_vec))
    
    # Comparison and Reporting
    for name, duration, iters, vec in results:
        print("\n" + "#"*60)
        print(f"  METHOD: {name}")
        print(f"  Execution Time: {duration:.6f} seconds")
        print(f"  Iterations    : {iters}")
        print("#"*60)
        print(f"\n  Top {args.top_k} Ranking:")
        print(f"  {'-'*35}")
        print(f"  {'Rank':<8} | {'Node ID':<10} | {'Score':<10}")
        print(f"  {'-'*35}")
        top_indices = np.argsort(vec)[::-1][:args.top_k]
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank:<8} | {idx+1:<10} | {vec[idx]:.6f}")
        print(f"  {'-'*35}")
        if len(vec) > 10:
            formatted_vec = "[" + ", ".join([f"{v:.4f}" for v in vec[:4]]) + " ... " + ", ".join([f"{v:.4f}" for v in vec[-4:]]) + "]"
        else:
            formatted_vec = vec
        print(f"\n  Vector Preview:\n  {formatted_vec}\n")
    print("\n" + "="*60)
    print(f"[*] Check ||Cold - Warm||_1: {np.linalg.norm(x_cold - x_warm, 1):.4e}")
    print(f"[*] Check ||Cold - Local||_1 : {np.linalg.norm(x_cold - x_vec, 1):.4e}")
    print("-" * 60)
    print(f"[*] Warm-Start Speedup: {t_cold / t_warm:.2f}x")
    print(f"[*] local Push Speedup: {t_cold / t_vec:.2f}x")
    print("="*60)
    print(f"{'END OF TEST':^60}")
    print("="*60)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_file_path', type=str, default='./data/Wiki-Vote.txt')
    argparser.add_argument('--p', type=float, default=0.85)
    argparser.add_argument('--tol', type=float, default=1e-9)
    argparser.add_argument('--top_k', type=int, default=5, help='Number of top nodes to display (default 10)')
    argparser.add_argument('--add_ratio', type=float, default=0.0001, help='Ratio of edges to add (default 0.01%)')
    argparser.add_argument('--drop_ratio', type=float, default=0.0001, help='Ratio of edges to drop (default 0.01%)')
    args = argparser.parse_args()
    
    main(args)