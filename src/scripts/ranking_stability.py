import os
import time
import json
import argparse
import numpy as np
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt

from static_textook_implementation_power_method import load_graph, build_components
from warm_start import perturb_graph, power_method_with_init_vec
from local_push import vectorized_local_push


def ranking_stability_metrics(x_old, x_new, top_k_list=None):
    if top_k_list is None:
        top_k_list = [10, 50, 100]

    tau, _ = kendalltau(x_old, x_new)
    rho, _ = spearmanr(x_old, x_new)

    rank_old = np.argsort(x_old)[::-1]
    rank_new = np.argsort(x_new)[::-1]

    metrics = {
        'kendall_tau': tau,
        'spearman_rho': rho,
    }
    for k in top_k_list:
        top_old = set(rank_old[:k])
        top_new = set(rank_new[:k])
        intersection = len(top_old & top_new)
        union = len(top_old | top_new)
        metrics[f'jaccard_top{k}'] = intersection / union if union > 0 else 0.0
        metrics[f'overlap_top{k}'] = intersection / k

    top_k_disp = []
    for k in (top_k_list[0], top_k_list[-1]):
        top_old_set = set(rank_old[:k])
        top_new_set = set(rank_new[:k])
        common = top_old_set & top_new_set
        shifts = []
        for node in common:
            shift = abs(np.where(rank_old == node)[0][0] - np.where(rank_new == node)[0][0])
            shifts.append(shift)
        if shifts:
            top_k_disp.append({
                'k': k,
                'max_shift': int(max(shifts)),
                'mean_shift': float(np.mean(shifts)),
            })
    metrics['top_k_displacement'] = top_k_disp

    return metrics


def run_experiments(data_file_path, p, tol, perturbation_ratios, top_k_list):
    edges, n_nodes = load_graph(data_file_path)

    G, D, e, f, _ = build_components(edges, n_nodes, p)
    print(f"[*] Computing original PageRank vector (n={n_nodes}) ...")
    x_old, _ = power_method_with_init_vec(G, D, e, f, n_nodes, p, tol=tol)

    print("-" * 60)

    results = []
    for ratio in perturbation_ratios:
        pct = ratio * 100
        print(f"\n[*] Perturbation ratio = {ratio:.4f} ({pct:.2f}%)")
        edges_new, added, dropped = perturb_graph(edges, n_nodes, add_ratio=ratio, drop_ratio=ratio)
        G_new, D_new, e_new, f_new, _ = build_components(edges_new, n_nodes, p)

        start = time.perf_counter()
        x_cold, iters_cold = power_method_with_init_vec(G_new, D_new, e_new, f_new, n_nodes, p, tol=tol)
        t_cold = time.perf_counter() - start

        start = time.perf_counter()
        x_warm, iters_warm = power_method_with_init_vec(G_new, D_new, e_new, f_new, n_nodes, p, x_init=x_old, tol=tol)
        t_warm = time.perf_counter() - start

        start = time.perf_counter()
        x_local, iters_local = vectorized_local_push(G_new, D_new, f_new, x_old, n_nodes, p, tol=tol)
        t_local = time.perf_counter() - start

        l1_old_cold = np.linalg.norm(x_old - x_cold, 1)
        l1_old_warm = np.linalg.norm(x_old - x_warm, 1)
        l1_old_local = np.linalg.norm(x_old - x_local, 1)

        l1_cold_warm = np.linalg.norm(x_cold - x_warm, 1)
        l1_cold_local = np.linalg.norm(x_cold - x_local, 1)

        stability_cold = ranking_stability_metrics(x_old, x_cold, top_k_list)
        stability_warm = ranking_stability_metrics(x_old, x_warm, top_k_list)
        stability_local = ranking_stability_metrics(x_old, x_local, top_k_list)

        stability_cw = ranking_stability_metrics(x_cold, x_warm, top_k_list)
        stability_cl = ranking_stability_metrics(x_cold, x_local, top_k_list)

        result = {
            'ratio': ratio,
            'perturbation_percent': f"{pct:.2f}%",
            'added_edges': added,
            'dropped_edges': dropped,
            'cold_iters': iters_cold,
            'warm_iters': iters_warm,
            'local_iters': iters_local,
            'l1_old_cold': l1_old_cold,
            'l1_old_warm': l1_old_warm,
            'l1_old_local': l1_old_local,
            'l1_cold_warm': l1_cold_warm,
            'l1_cold_local': l1_cold_local,
            'stability_cold': stability_cold,
            'stability_warm': stability_warm,
            'stability_local': stability_local,
            'stability_cw': stability_cw,
            'stability_cl': stability_cl,
        }
        results.append(result)

        print(f"    ||x_old - x_cold||_1 = {l1_old_cold:.6e}")
        print(f"    Kendall tau (old vs cold)  = {stability_cold['kendall_tau']:.6f}")
        print(f"    Top-10 Jaccard (old vs cold) = {stability_cold['jaccard_top10']:.4f}")
        print(f"    Top-100 Jaccard (old vs cold) = {stability_cold['jaccard_top100']:.4f}")

    return results, edges, n_nodes


def plot_results(results, save_path, top_k_list):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ratios = [r['ratio'] for r in results]
    labels = [r['perturbation_percent'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ---- Plot 1: Ranking correlation metrics ----
    ax1 = axes[0, 0]
    ax1.plot(ratios, [r['stability_cold']['kendall_tau'] for r in results],
             'o-', color='#2c7bb6', label=r"Kendall $\tau$", markersize=7)
    ax1.plot(ratios, [r['stability_cold']['spearman_rho'] for r in results],
             's--', color='#fdae61', label=r"Spearman $\rho$", markersize=7)
    ax1.set_xscale('log')
    ax1.set_xlabel('Perturbation Ratio')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Ranking Correlation (x_old vs x_cold)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.85, 1.02)

    # ---- Plot 2: Top-k Jaccard ----
    ax2 = axes[0, 1]
    colors = ['#d7191c', '#2b83ba', '#abdda4']
    for i, k in enumerate(top_k_list):
        ax2.plot(ratios, [r['stability_cold'][f'jaccard_top{k}'] for r in results],
                 'o-' if i == 0 else 's--' if i == 1 else '^:',
                 color=colors[i], label=f'Top-{k}', markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Perturbation Ratio')
    ax2.set_ylabel('Jaccard Similarity')
    ax2.set_title('Top-k Jaccard (x_old vs x_cold)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # ---- Plot 3: Top-k Overlap rate ----
    ax3 = axes[0, 2]
    for i, k in enumerate(top_k_list):
        ax3.plot(ratios, [r['stability_cold'][f'overlap_top{k}'] for r in results],
                 'o-' if i == 0 else 's--' if i == 1 else '^:',
                 color=colors[i], label=f'Top-{k}', markersize=6)
    ax3.set_xscale('log')
    ax3.set_xlabel('Perturbation Ratio')
    ax3.set_ylabel('Overlap Rate')
    ax3.set_title('Top-k Overlap Rate (x_old vs x_cold)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # ---- Plot 4: L1 norm vs perturbation ----
    ax4 = axes[1, 0]
    ax4.plot(ratios, [r['l1_old_cold'] for r in results],
             'o-', color='#d73027', label=r'$||x_{old} - x_{cold}||_1$', markersize=7)
    ax4.plot(ratios, [r['l1_old_warm'] for r in results],
             's--', color='#fc8d59', label=r'$||x_{old} - x_{warm}||_1$', markersize=7, alpha=0.7)
    ax4.plot(ratios, [r['l1_old_local'] for r in results],
             '^:', color='#91bfdb', label=r'$||x_{old} - x_{local}||_1$', markersize=7, alpha=0.7)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Perturbation Ratio')
    ax4.set_ylabel('L1 Norm')
    ax4.set_title('PageRank Vector L1 Change')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ---- Plot 5: L1 norm vs Kendall tau (scatter with labels) ----
    ax5 = axes[1, 1]
    l1_values = [r['l1_old_cold'] for r in results]
    tau_values = [r['stability_cold']['kendall_tau'] for r in results]
    ax5.scatter(l1_values, tau_values, c='#2c7bb6', s=100, edgecolors='black', zorder=5)
    for i, r in enumerate(results):
        ax5.annotate(r['perturbation_percent'], (l1_values[i], tau_values[i]),
                     textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax5.set_xscale('log')
    ax5.set_xlabel(r'$||x_{old} - x_{cold}||_1$')
    ax5.set_ylabel(r"Kendall $\tau$")
    ax5.set_title('L1 Error vs Ranking Stability')
    ax5.grid(True, alpha=0.3)
    l1_min, l1_max = min(l1_values), max(l1_values)
    margin = 0.1 * (np.log10(l1_max) - np.log10(l1_min))
    ax5.set_xlim(10**(np.log10(l1_min) - margin), 10**(np.log10(l1_max) + margin))

    # ---- Plot 6: Jaccard Top-10 vs L1 norm ----
    ax6 = axes[1, 2]
    jaccard10_values = [r['stability_cold']['jaccard_top10'] for r in results]
    for i, r in enumerate(results):
        ax6.plot(l1_values[i], jaccard10_values[i], 'o', color='#d7191c',
                 markersize=10, label='' if i > 0 else 'Top-10 Jaccard')
    ax6.set_xscale('log')
    ax6.set_xlabel(r'$||x_{old} - x_{cold}||_1$')
    ax6.set_ylabel('Top-10 Jaccard')
    ax6.set_title('L1 Error vs Top-10 Overlap')
    ax6.annotate(labels[0], (l1_values[0], jaccard10_values[0]),
                 textcoords="offset points", xytext=(8, -8), fontsize=8)
    ax6.annotate(labels[-1], (l1_values[-1], jaccard10_values[-1]),
                 textcoords="offset points", xytext=(8, -8), fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1.05)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[*] Plot saved to {save_path}")


def save_json(results, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    serializable = []
    for r in results:
        item = {k: v for k, v in r.items()}
        item['l1_old_cold'] = float(item['l1_old_cold'])
        item['l1_old_warm'] = float(item['l1_old_warm'])
        item['l1_old_local'] = float(item['l1_old_local'])
        item['l1_cold_warm'] = float(item['l1_cold_warm'])
        item['l1_cold_local'] = float(item['l1_cold_local'])
        for key in ('stability_cold', 'stability_warm', 'stability_local',
                     'stability_cw', 'stability_cl'):
            if key in item:
                item[key] = {k: (float(v) if isinstance(v, (np.integer, np.floating))
                                  else v) for k, v in item[key].items()}
        serializable.append(item)
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"[*] JSON data saved to {json_path}")


def main(args):
    top_k_list = args.top_k_list
    perturbation_ratios = args.ratios

    print("=" * 60)
    print(f"{'TOP-K RANKING STABILITY ANALYSIS':^60}")
    print("=" * 60)
    print(f"[*] Data   : {args.data_file_path}")
    print(f"[*] p      : {args.p}")
    print(f"[*] tol    : {args.tol}")
    print(f"[*] Ratios : {perturbation_ratios}")
    print(f"[*] Top-k  : {top_k_list}")
    print("-" * 60)

    results, edges, n_nodes = run_experiments(
        args.data_file_path, args.p, args.tol, perturbation_ratios, top_k_list
    )

    print("\n" + "=" * 60)
    print(f"{'RANKING STABILITY SUMMARY':^60}")
    print("=" * 60)
    header = f"{'Ratio':>10s} | {'L1(x_old-cold)':>16s} | {'Kendall tau':>12s} | {'Jaccard T10':>12s} | {'Jaccard T100':>12s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['perturbation_percent']:>10s} | {r['l1_old_cold']:>16.6e} | "
              f"{r['stability_cold']['kendall_tau']:>12.6f} | "
              f"{r['stability_cold']['jaccard_top10']:>12.4f} | "
              f"{r['stability_cold']['jaccard_top100']:>12.4f}")

    top_k_disp = results[-1]['stability_cold'].get('top_k_displacement', [])
    if top_k_disp:
        print("\n[*] Max displacement at max perturbation:")
        for d in top_k_disp:
            print(f"    Top-{d['k']}: max_shift={d['max_shift']}, mean_shift={d['mean_shift']:.2f}")

    print("\n[*] Cold vs Warm-Start consistency:")
    print(f"    max ||x_cold - x_warm||_1 = {max(r['l1_cold_warm'] for r in results):.4e}")
    print(f"    min Kendall tau (cold vs warm) = {min(r['stability_cw']['kendall_tau'] for r in results):.6f}")

    print("\n[*] Cold vs Local Push consistency:")
    print(f"    max ||x_cold - x_local||_1 = {max(r['l1_cold_local'] for r in results):.4e}")
    print(f"    min Kendall tau (cold vs local) = {min(r['stability_cl']['kendall_tau'] for r in results):.6f}")

    plot_results(results, args.output_path, top_k_list)
    save_json(results, args.json_path)
    print("=" * 60)
    print(f"{'END OF ANALYSIS':^60}")
    print("=" * 60)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Top-k Ranking Stability Analysis for Incremental PageRank'
    )
    argparser.add_argument('--data_file_path', type=str, default='./data/Wiki-Vote.txt',
                           help='Path to the graph data file')
    argparser.add_argument('--p', type=float, default=0.85,
                           help='Damping factor')
    argparser.add_argument('--tol', type=float, default=1e-9,
                           help='Convergence tolerance')
    argparser.add_argument('--ratios', type=float, nargs='+',
                           default=[0.001, 0.005, 0.01, 0.05, 0.1],
                           help='Perturbation ratios (add and drop)')
    argparser.add_argument('--top_k_list', type=int, nargs='+',
                           default=[10, 50, 100],
                           help='List of k values for top-k analysis')
    argparser.add_argument('--output_path', type=str,
                           default='./results/ranking_stability.png',
                           help='Path to save the output plot')
    argparser.add_argument('--json_path', type=str,
                           default='./results/ranking_stability.json',
                           help='Path to save JSON results')
    args = argparser.parse_args()
    main(args)
