# Theoretical bound for PageRank perturbation: ||Δx||₁ ≤ (p / (1 - p)) * ||ΔM||₁
import os
import argparse
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from static_textook_implementation_power_method import (
    load_graph,
    build_components,
    power_method_standard as power_method
)


def compute_delta_M_norm1(G_old, D_old, G_new, D_new, n_nodes):
    """
    Computes the L1 norm of ΔM_bar, where M_bar is the column-stochastic transition matrix adjusted for dangling nodes:
    M_bar = G * D + (1/n) * e * d^T
    """
    # Determine dangling node indicators (1 if dangling, 0 otherwise)
    d_old = (np.array(G_old.sum(axis=0)).flatten() == 0).astype(float)
    d_new = (np.array(G_new.sum(axis=0)).flatten() == 0).astype(float)
    # Calculate sparse part: M_sparse = G * D
    # We use csc_matrix to efficiently extract and operate column by column
    M_old_sparse = (G_old @ sp.diags(D_old)).tocsc()
    M_new_sparse = (G_new @ sp.diags(D_new)).tocsc()
    max_norm = 0.0
    for j in range(n_nodes):
        # Difference in the sparse topology part for column j
        diff_sparse = M_new_sparse[:, j] - M_old_sparse[:, j]
        d_diff = d_new[j] - d_old[j]
        if d_diff == 0:
            # If the dangling status hasn't changed, the uniform compensation cancels out.
            # We can compute the L1 norm purely on the sparse non-zero elements.
            col_norm = np.abs(diff_sparse.data).sum()
        else:
            # If dangling status changed, we must compute the norm using the dense vector format
            # due to the dense uniform probability injection (or removal).
            diff_dense = diff_sparse.toarray().flatten()
            diff_dense += (1.0 / n_nodes) * d_diff
            col_norm = np.abs(diff_dense).sum()
        if col_norm > max_norm:
            max_norm = col_norm
    return max_norm


def simulate_perturbation(edges, n_nodes, p, add_ratio, drop_ratio):
    # Original graph components and PageRank
    G, D, e, f, v = build_components(edges, n_nodes, p)
    x_old = power_method(G, D, e, f, n_nodes, p, max_iter=200, tol=1e-9)
    
    # -----------------------------
    # Perturbation Simulation
    # -----------------------------
    n_edges = len(edges)
    # a. Delete edges randomly based on drop_ratio
    drop_count = int(n_edges * drop_ratio)
    keep_indices = np.random.choice(n_edges, n_edges - drop_count, replace=False)
    edges_kept = edges[keep_indices]
    # b. Add edges randomly based on add_ratio
    add_count = int(n_edges * add_ratio)
    new_edges = np.random.randint(0, n_nodes, size=(add_count, 2))
    edges_new = np.vstack([edges_kept, new_edges])
    # -----------------------------

    # Perturbed graph components and PageRank
    G_new, D_new, e_new, f_new, v_new = build_components(edges_new, n_nodes, p)
    x_new = power_method(G_new, D_new, e_new, f_new, n_nodes, p, max_iter=200, tol=1e-9)

    # Calculate L1 Norm of Vector Perturbation (Δx)
    delta_x = x_new - x_old
    delta_x_norm1 = np.linalg.norm(delta_x, 1)    
    
    # Calculate L1 Norm of the column-stochastic matrix Perturbation (ΔM_bar)
    delta_M_norm1 = compute_delta_M_norm1(G, D, G_new, D_new, n_nodes)
    
    # The absolute theoretical bound upper limit
    theoretical_bound = (p / (1 - p)) * delta_M_norm1
    is_bounded = delta_x_norm1 <= theoretical_bound + 1e-12
    return delta_x_norm1, delta_M_norm1, theoretical_bound, is_bounded


def main(data_file_path, p, add_ratio, drop_ratio):
    edges, n_nodes = load_graph(data_file_path)
    delta_x_norm1, delta_M_norm1, theoretical_bound, is_bounded = simulate_perturbation(
        edges, n_nodes, p, add_ratio, drop_ratio
    )
    print("Matrix Perturbation Simulation Results:")
    print(f"||Δx||₁: {delta_x_norm1:.6f}")
    print(f"||ΔM||₁: {delta_M_norm1:.6f}")
    print(f"Theoretical Bound: ||Δx||₁ ≤ (p / (1 - p)) * ||ΔM||₁ = {theoretical_bound:.6f}")
    print(f"Is ||Δx||₁ ≤ Theoretical Bound? {'Yes' if is_bounded else 'No'}")


def run_experiments_and_plot(data_file_path, p, add_ratios, drop_ratios, variable, result_save_path):
    edges, n_nodes = load_graph(data_file_path)
    delta_x_norm1_list = []
    delta_M_norm1_list = []
    theoretical_bound_list = []
    
    for add_ratio in add_ratios if variable == 'add_ratio' else [add_ratios]:
        for drop_ratio in drop_ratios if variable == 'drop_ratio' else [drop_ratios]:
            delta_x_norm1, delta_M_norm1, theoretical_bound, _ = simulate_perturbation(
                edges, n_nodes, p, add_ratio, drop_ratio
            )
            delta_x_norm1_list.append(delta_x_norm1)
            delta_M_norm1_list.append(delta_M_norm1)
            theoretical_bound_list.append(theoretical_bound)

    # Plotting
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    x_values = add_ratios if variable == 'add_ratio' else drop_ratios
    plt.plot(x_values, delta_x_norm1_list, marker='o', label='||Δx||₁')
    plt.plot(x_values, theoretical_bound_list, marker='x', label='Theoretical Bound (p/(1-p)*||ΔM||₁)')
    plt.xlabel(variable.replace('_', ' ').title())
    plt.ylabel('L1 Norm')
    plt.title(f'Matrix Perturbation Simulation: Varying {variable.replace("_", " ").title()}')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(result_save_path)
    print(f"Experiment results plotted and saved to {result_save_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Matrix Perturbation Simulation for PageRank')
    argparser.add_argument('--data_file_path', type=str, default='./data/Wiki-Vote.txt', help='Path to the data file')
    argparser.add_argument('--p', type=float, default=0.85, help='Damping factor')
    argparser.add_argument('--add_ratio', type=float, default=0.001, help='Ratio of edges to add')
    argparser.add_argument('--drop_ratio', type=float, default=0.001, help='Ratio of edges to drop')
    argparser.add_argument('--run_experiments', action='store_true', help='Run multi-ratio experiments and plot')
    argparser.add_argument('--variable', type=str, choices=['add_ratio', 'drop_ratio'], default='drop_ratio', help='Variable to vary in experiments')
    args = argparser.parse_args()

    if args.run_experiments:
        if args.variable == 'add_ratio':
            add_ratios = [0.001, 0.005, 0.01, 0.05, 0.1]
            run_experiments_and_plot(args.data_file_path, args.p, add_ratios, args.drop_ratio, 'add_ratio', './results/matrix_perturbation_add_ratio_results.png')
        else:
            drop_ratios = [0.001, 0.005, 0.01, 0.05, 0.1]
            run_experiments_and_plot(args.data_file_path, args.p, args.add_ratio, drop_ratios, 'drop_ratio', './results/matrix_perturbation_drop_ratio_results.png')
    else:
        main(args.data_file_path, args.p, args.add_ratio, args.drop_ratio)