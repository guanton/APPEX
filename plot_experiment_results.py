import os
import re
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.tree._criterion import MAE


# ===============================
# Helper: Get Minimum Indices
# ===============================
def get_min_indices(convergence_scores):
    """
    Given a list of dictionaries (one per iteration) with possible keys among 'nll', 'w2', 'w1', 'mmd',
    return a dictionary with the iteration index (0-indexed) of the minimum value for each metric that exists.
    """
    min_indices = {}
    possible_keys = ['nll', 'w2', 'w1', 'mmd']
    for key in possible_keys:
        if any(key in score for score in convergence_scores):
            # For missing entries use np.inf so that they are not chosen as minimum
            values = [score.get(key, np.inf) for score in convergence_scores]
            if not all(val == np.inf for val in values):
                min_indices[key] = int(np.argmin(values))
    return min_indices


# ===============================
# Basic Metrics
# ===============================
def compute_mae(estimated, ground_truth):
    """Compute Mean Absolute Error (MAE)."""
    return np.mean(np.abs(estimated - ground_truth))


def calculate_correlation(estimated_matrix, ground_truth_matrix):
    """Compute Pearson correlation between flattened matrices."""
    est_flat = estimated_matrix.flatten()
    gt_flat = ground_truth_matrix.flatten()
    return np.corrcoef(est_flat, gt_flat)[0, 1]


# ===============================
# SHD Functions
# ===============================
def compute_shd(true_matrix, est_matrix, edge_threshold):
    """
    Compute the Structural Hamming Distance (SHD) for direct edges.
    An edge is considered present if abs(value) > edge_threshold.
    """
    d = true_matrix.shape[0]
    true_edges = {(i, j) for i in range(d) for j in range(d)
                  if abs(true_matrix[j, i]) > edge_threshold}
    est_edges = {(i, j) for i in range(d) for j in range(d)
                 if abs(est_matrix[j, i]) > edge_threshold}
    return len(true_edges.symmetric_difference(est_edges))


def compute_v_structure_shd(H_true, H_est, edge_threshold=1.0):
    """
    Compute the Structural Hamming Distance (SHD) for v-structures between two matrices.
    """
    d = H_true.shape[0]
    shd_v_structure = 0
    for i in range(d):
        for j in range(d):
            if i != j:
                if abs(H_true[i, j]) >= edge_threshold and abs(H_est[i, j]) < edge_threshold:
                    shd_v_structure += 1
                elif abs(H_true[i, j]) < edge_threshold and abs(H_est[i, j]) >= edge_threshold:
                    shd_v_structure += 1
    return shd_v_structure / 2


# ===============================
# Graph Construction and Layout
# ===============================
def construct_causal_graph(A, H, edge_threshold=0.5, v_threshold=0.5):
    d = A.shape[0]
    G = nx.DiGraph()
    main_nodes = list(range(1, d + 1))
    G.add_nodes_from(main_nodes)
    for i in range(d):
        for j in range(d):
            if abs(A[j, i]) > edge_threshold:
                color = 'red' if A[j, i] < 0 else 'green'
                G.add_edge(i + 1, j + 1, color=color, curved=False, style='solid')
    for i in range(d):
        for j in range(i + 1, d):
            if abs(H[i, j]) >= v_threshold:
                exog = f"U_{i + 1}_{j + 1}"
                G.add_node(exog)
                G.add_edge(exog, i + 1, color='black', style='dotted')
                G.add_edge(exog, j + 1, color='black', style='dotted')
    return G


def compute_layout(graph, offset=0.3):
    main_nodes = [n for n in graph.nodes() if isinstance(n, int)]
    pos = nx.circular_layout(main_nodes)
    for node in graph.nodes():
        if not isinstance(node, int):
            m = re.match(r"U_(\d+)_(\d+)", node)
            if m:
                i, j = int(m.group(1)), int(m.group(2))
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length == 0:
                    length = 1
                pos[node] = (mid_x + (dy / length) * offset, mid_y - (dx / length) * offset)
    return pos


def plot_single_graph(graph, pos, title, ax):
    """
    Draw the graph with the given layout and title.
    """
    filtered_edges = {}
    for u, v, data in graph.edges(data=True):
        key = (u, v)
        if key not in filtered_edges:
            filtered_edges[key] = data
        else:
            if data.get('color') == 'red':
                filtered_edges[key] = data
    straight_edges = [(u, v) for (u, v), data in filtered_edges.items() if not data.get('curved', False)]
    curved_edges = [(u, v) for (u, v), data in filtered_edges.items() if data.get('curved', False)]
    dotted_edges = [(u, v) for (u, v), data in filtered_edges.items() if data.get('style') == 'dotted']
    if straight_edges:
        edge_colors = [filtered_edges[(u, v)]['color'] for (u, v) in straight_edges]
    else:
        edge_colors = ['black']
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=800, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', ax=ax)
    if straight_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=straight_edges, edge_color=edge_colors,
                               arrows=True, arrowstyle='-|>', arrowsize=8, ax=ax)
    if curved_edges:
        edge_colors_curve = [filtered_edges[(u, v)]['color'] for (u, v) in curved_edges]
        nx.draw_networkx_edges(graph, pos, edgelist=curved_edges, edge_color=edge_colors_curve,
                               arrows=True, arrowstyle='-|>', arrowsize=8,
                               connectionstyle='arc3,rad=0.3', ax=ax)
    if dotted_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=dotted_edges, edge_color='black',
                               arrows=True, arrowstyle='-|>', arrowsize=8, ax=ax,
                               style=(0, (15, 10)))
    ax.set_title(title)
    ax.set_aspect('equal')


def plot_MAEs(iterations, A_maes, H_maes, min_indices, exp_title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, A_maes, label='MAE (A)', color='black', marker='o')
    plt.plot(iterations, H_maes, label='MAE (H)', color='gray', marker='o')
    # Only add vertical lines for metrics that are present.
    metric_colors = {'nll': 'red', 'w2': 'blue', 'w1': 'green', 'mmd': 'orange'}
    for metric in ['nll', 'w2', 'w1', 'mmd']:
        if metric in min_indices:
            plt.axvline(x=min_indices[metric] + 1, color=metric_colors[metric],
                        linestyle='--', label=f'Min {metric.upper()}')
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.title(exp_title if exp_title else 'MAE vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation(iterations, A_corrs, H_corrs, min_indices, exp_title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, A_corrs, label='Corr (A)', color='black', marker='o')
    plt.plot(iterations, H_corrs, label='Corr (H)', color='gray', marker='o')
    metric_colors = {'nll': 'red', 'w2': 'blue', 'w1': 'green', 'mmd': 'orange'}
    for metric in ['nll', 'w2', 'w1', 'mmd']:
        if metric in min_indices:
            plt.axvline(x=min_indices[metric] + 1, color=metric_colors[metric],
                        linestyle='--', label=f'Min {metric.upper()}')
    plt.xlabel('Iteration')
    plt.ylabel('Correlation')
    plt.title(exp_title if exp_title else 'Correlation vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_normalized_convergence(results_data, exp_title=None):
    """
    Generate a single normalized convergence plot with available metrics together.
    """
    num_iterations = len(results_data['convergence scores'])
    iterations = np.arange(1, num_iterations + 1)
    conv_scores = results_data['convergence scores']
    metric_colors = {'nll': 'red', 'w2': 'blue', 'w1': 'green', 'mmd': 'orange'}

    plt.figure(figsize=(10, 6))
    for metric in ['nll', 'w2', 'w1', 'mmd']:
        # Only plot if every iteration has the metric
        if all(metric in score for score in conv_scores):
            metric_vals = np.array([score[metric] for score in conv_scores])
            metric_min, metric_max = np.min(metric_vals), np.max(metric_vals)
            normalized_vals = (metric_vals - metric_min) / (
                    metric_max - metric_min) if metric_max != metric_min else np.zeros_like(metric_vals)
            min_index = int(np.argmin(metric_vals))
            plt.plot(iterations, normalized_vals, label=f'Normalized {metric.upper()}',
                     color=metric_colors[metric], marker='o')
            plt.axvline(x=min_index + 1, color=metric_colors[metric],
                        linestyle='--', label=f'Min {metric.upper()}')
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Convergence Score')
    plt.title(exp_title if exp_title else 'Normalized Convergence Metrics vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===============================
# Aggregate Selection Metrics
# ===============================
def aggregate_selection_metrics(replicates, edge_threshold=0.5, v_threshold=1, verbose=False):
    """
    For each replicate, extract the following metrics (using the replicate's ground truth):
      - WOT: metrics at iteration 0.
      - APPEX: metrics at the last iteration.
      - Best iteration by available convergence metrics (nll, w1, w2, mmd) if they exist.
    For each replicate and each selection, compute:
      * MAE for A and H
      * Pearson correlation for A and H
      * Direct SHD for A and latent SHD for H (using v_threshold for H)
    Then, aggregate across replicates by computing the mean and standard error for each metric.
    """
    selections = {
        "WOT": {"MAE_A": [], "MAE_H": [], "Corr_A": [], "Corr_H": [], "SHD_A": [], "vSHD_H": []},
        "APPEX": {"MAE_A": [], "MAE_H": [], "Corr_A": [], "Corr_H": [], "SHD_A": [], "vSHD_H": []},
        "min_nll": {"MAE_A": [], "MAE_H": [], "Corr_A": [], "Corr_H": [], "SHD_A": [], "vSHD_H": []},
        "min_w1": {"MAE_A": [], "MAE_H": [], "Corr_A": [], "Corr_H": [], "SHD_A": [], "vSHD_H": []},
        "min_w2": {"MAE_A": [], "MAE_H": [], "Corr_A": [], "Corr_H": [], "SHD_A": [], "vSHD_H": []},
        "min_mmd": {"MAE_A": [], "MAE_H": [], "Corr_A": [], "Corr_H": [], "SHD_A": [], "vSHD_H": []}
    }

    for key in sorted(replicates.keys()):
        rep_data = replicates[key]
        true_A = rep_data['true_A']
        true_H = rep_data['true_H']
        if verbose:
            print(f"\n--- Replicate {key} ---")
            print("Ground Truth A:")
            print(true_A)
            print("Ground Truth H:")
            print(true_H)
        num_iterations = len(rep_data['est A values'])

        # WOT: iteration 0
        idx_wot = 0
        A_wot = rep_data['est A values'][idx_wot]
        H_wot = rep_data['est H values'][idx_wot]
        selections["WOT"]["MAE_A"].append(compute_mae(A_wot, true_A))
        selections["WOT"]["MAE_H"].append(compute_mae(H_wot, true_H))
        selections["WOT"]["Corr_A"].append(calculate_correlation(A_wot, true_A))
        selections["WOT"]["Corr_H"].append(calculate_correlation(H_wot, true_H))
        selections["WOT"]["SHD_A"].append(compute_shd(A_wot, true_A, edge_threshold))
        selections["WOT"]["vSHD_H"].append(compute_v_structure_shd(true_H, H_wot, v_threshold))

        # APPEX: last iteration
        idx_appex = num_iterations - 1
        A_appex = rep_data['est A values'][idx_appex]
        H_appex = rep_data['est H values'][idx_appex]
        selections["APPEX"]["MAE_A"].append(compute_mae(A_appex, true_A))
        selections["APPEX"]["MAE_H"].append(compute_mae(H_appex, true_H))
        selections["APPEX"]["Corr_A"].append(calculate_correlation(A_appex, true_A))
        selections["APPEX"]["Corr_H"].append(calculate_correlation(H_appex, true_H))
        selections["APPEX"]["SHD_A"].append(compute_shd(A_appex, true_A, edge_threshold))
        selections["APPEX"]["vSHD_H"].append(compute_v_structure_shd(true_H, H_appex, v_threshold))

        # Best iterations per replicate for available convergence metrics:
        conv_scores = rep_data['convergence scores']

        if any('nll' in score for score in conv_scores):
            idx_min_nll = int(np.argmin([score.get('nll', np.inf) for score in conv_scores]))
            A_min_nll = rep_data['est A values'][idx_min_nll]
            H_min_nll = rep_data['est H values'][idx_min_nll]
            selections["min_nll"]["MAE_A"].append(compute_mae(A_min_nll, true_A))
            selections["min_nll"]["MAE_H"].append(compute_mae(H_min_nll, true_H))
            selections["min_nll"]["Corr_A"].append(calculate_correlation(A_min_nll, true_A))
            selections["min_nll"]["Corr_H"].append(calculate_correlation(H_min_nll, true_H))
            selections["min_nll"]["SHD_A"].append(compute_shd(A_min_nll, true_A, edge_threshold))
            selections["min_nll"]["vSHD_H"].append(compute_v_structure_shd(true_H, H_min_nll, v_threshold))

        if any('w1' in score for score in conv_scores):
            idx_min_w1 = int(np.argmin([score.get('w1', np.inf) for score in conv_scores]))
            A_min_w1 = rep_data['est A values'][idx_min_w1]
            H_min_w1 = rep_data['est H values'][idx_min_w1]
            selections["min_w1"]["MAE_A"].append(compute_mae(A_min_w1, true_A))
            selections["min_w1"]["MAE_H"].append(compute_mae(H_min_w1, true_H))
            selections["min_w1"]["Corr_A"].append(calculate_correlation(A_min_w1, true_A))
            selections["min_w1"]["Corr_H"].append(calculate_correlation(H_min_w1, true_H))
            selections["min_w1"]["SHD_A"].append(compute_shd(A_min_w1, true_A, edge_threshold))
            selections["min_w1"]["vSHD_H"].append(compute_v_structure_shd(true_H, H_min_w1, v_threshold))

        if any('w2' in score for score in conv_scores):
            idx_min_w2 = int(np.argmin([score.get('w2', np.inf) for score in conv_scores]))
            A_min_w2 = rep_data['est A values'][idx_min_w2]
            H_min_w2 = rep_data['est H values'][idx_min_w2]
            selections["min_w2"]["MAE_A"].append(compute_mae(A_min_w2, true_A))
            selections["min_w2"]["MAE_H"].append(compute_mae(H_min_w2, true_H))
            selections["min_w2"]["Corr_A"].append(calculate_correlation(A_min_w2, true_A))
            selections["min_w2"]["Corr_H"].append(calculate_correlation(H_min_w2, true_H))
            selections["min_w2"]["SHD_A"].append(compute_shd(A_min_w2, true_A, edge_threshold))
            selections["min_w2"]["vSHD_H"].append(compute_v_structure_shd(true_H, H_min_w2, v_threshold))

        if any('mmd' in score for score in conv_scores):
            idx_min_mmd = int(np.argmin([score.get('mmd', np.inf) for score in conv_scores]))
            A_min_mmd = rep_data['est A values'][idx_min_mmd]
            H_min_mmd = rep_data['est H values'][idx_min_mmd]
            selections["min_mmd"]["MAE_A"].append(compute_mae(A_min_mmd, true_A))
            selections["min_mmd"]["MAE_H"].append(compute_mae(H_min_mmd, true_H))
            selections["min_mmd"]["Corr_A"].append(calculate_correlation(A_min_mmd, true_A))
            selections["min_mmd"]["Corr_H"].append(calculate_correlation(H_min_mmd, true_H))
            selections["min_mmd"]["SHD_A"].append(compute_shd(A_min_mmd, true_A, edge_threshold))
            selections["min_mmd"]["vSHD_H"].append(compute_v_structure_shd(true_H, H_min_mmd, v_threshold))

    aggregated = {}
    for sel in selections:
        aggregated[sel] = {}
        for metric in selections[sel]:
            values = np.array(selections[sel][metric])
            if len(values) > 0:
                aggregated[sel][metric + '_mean'] = np.mean(values)
                aggregated[sel][metric + '_se'] = np.std(values, ddof=1) / np.sqrt(len(values)) if len(
                    values) > 1 else 0.0
            else:
                aggregated[sel][metric + '_mean'] = np.nan
                aggregated[sel][metric + '_se'] = np.nan
    return aggregated


# ===============================
# Causal Graph Plotting (2x3 Display)
# ===============================
def plot_six_causal_graphs(results_data, true_A, true_H, v_threshold=1, edge_threshold=0.5):
    """
    Display a 2 x 3 grid of causal graphs.
    Top row: True graph, Iteration 1 (WOT), and best iteration among available metrics (preferring mmd if available).
    Bottom row: Estimated graphs at iterations with minimum nll, minimum w2, and minimum w1 (if available).
    """
    est_A_vals = results_data['est A values']
    est_H_vals = results_data['est H values']
    conv_scores = results_data['convergence scores']
    min_indices = get_min_indices(conv_scores)

    g_true = construct_causal_graph(true_A, true_H, v_threshold=v_threshold, edge_threshold=edge_threshold)
    g_wot = construct_causal_graph(est_A_vals[0], est_H_vals[0], v_threshold=v_threshold, edge_threshold=edge_threshold)

    # For the top row best graph, use mmd if available; otherwise, choose the first available metric.
    if 'mmd' in min_indices:
        best_top_idx = min_indices['mmd']
        best_top_label = "Best (Min MMD)"
    elif min_indices:
        best_top_idx = list(min_indices.values())[0]
        best_top_label = f"Best ({list(min_indices.keys())[0].upper()})"
    else:
        best_top_idx = 0
        best_top_label = "APPEX"
    g_best_top = construct_causal_graph(est_A_vals[best_top_idx], est_H_vals[best_top_idx],
                                        v_threshold=v_threshold, edge_threshold=edge_threshold)

    # Bottom row: use available metrics, defaulting to WOT if not available.
    if 'nll' in min_indices:
        g_min_nll = construct_causal_graph(est_A_vals[min_indices['nll']], est_H_vals[min_indices['nll']],
                                           v_threshold=v_threshold, edge_threshold=edge_threshold)
    else:
        g_min_nll = g_wot
    if 'w2' in min_indices:
        g_min_w2 = construct_causal_graph(est_A_vals[min_indices['w2']], est_H_vals[min_indices['w2']],
                                          v_threshold=v_threshold, edge_threshold=edge_threshold)
    else:
        g_min_w2 = g_wot
    if 'w1' in min_indices:
        g_min_w1 = construct_causal_graph(est_A_vals[min_indices['w1']], est_H_vals[min_indices['w1']],
                                          v_threshold=v_threshold, edge_threshold=edge_threshold)
    else:
        g_min_w1 = g_wot

    pos_true = compute_layout(g_true)
    pos_wot = compute_layout(g_wot)
    pos_best_top = compute_layout(g_best_top)
    pos_min_nll = compute_layout(g_min_nll)
    pos_min_w2 = compute_layout(g_min_w2)
    pos_min_w1 = compute_layout(g_min_w1)

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    # Top row:
    plot_single_graph(g_true, pos_true, "True Causal Graph", axes[0, 0])
    plot_single_graph(g_wot, pos_wot, "Iteration 1 (WOT)", axes[0, 1])
    plot_single_graph(g_best_top, pos_best_top, best_top_label, axes[0, 2])
    # Bottom row:
    plot_single_graph(g_min_nll, pos_min_nll,
                      "Best (Min NLL)" if 'nll' in min_indices else "WOT", axes[1, 0])
    plot_single_graph(g_min_w2, pos_min_w2,
                      "Best (Min W2)" if 'w2' in min_indices else "WOT", axes[1, 1])
    plot_single_graph(g_min_w1, pos_min_w1,
                      "Best (Min W1)" if 'w1' in min_indices else "WOT", axes[1, 2])
    plt.tight_layout()
    plt.show()


# ===============================
# Replicate File Loading
# ===============================
def load_replicate_files(directory):
    replicates = {}
    for idx, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                replicates[idx] = pickle.load(f)
    return replicates


# ===============================
# Main Interpretation Function
# ===============================
def interpret_experiment_directory(directory, exp_title=None, display_plots=True,
                                   causal_plots=True, causality_only=False,
                                   edge_threshold=0.5, v_eps=1, verbose_graph=False):
    replicates = load_replicate_files(directory)
    num_reps = len(replicates)
    print(f"Found {num_reps} replicates in directory '{directory}'")
    for rep, results_data in replicates.items():
        true_A = results_data['true_A']
        true_H = results_data['true_H']
        if display_plots:
            plot_replicate_diagnostics(results_data, true_A, true_H,
                                       exp_title=f"{exp_title} - Replicate {rep}")
            if causal_plots:
                plot_six_causal_graphs(results_data, true_A, true_H, v_threshold=v_eps, edge_threshold=edge_threshold)

    # Aggregate and print overall metrics.
    agg_sel = aggregate_selection_metrics(replicates, edge_threshold=edge_threshold, v_threshold=v_eps)
    print(f'Using edge threshold {edge_threshold} and v-eps {v_eps}')
    print("\nAggregated Metrics (across replicates):")
    for sel in ["WOT", "APPEX", "min_nll", "min_w1", "min_w2", "min_mmd"]:
        print(f"\nSelection: {sel}")
        if not causality_only:
            print("  Mean MAE for A: {:.4f} ± {:.4f}".format(agg_sel[sel]['MAE_A_mean'], agg_sel[sel]['MAE_A_se']))
            print("  Mean MAE for H: {:.4f} ± {:.4f}".format(agg_sel[sel]['MAE_H_mean'], agg_sel[sel]['MAE_H_se']))
            print("  Mean Corr for A: {:.4f} ± {:.4f}".format(agg_sel[sel]['Corr_A_mean'], agg_sel[sel]['Corr_A_se']))
            print("  Mean Corr for H: {:.4f} ± {:.4f}".format(agg_sel[sel]['Corr_H_mean'], agg_sel[sel]['Corr_H_se']))
        print("  Mean Direct SHD for A: {:.4f} ± {:.4f}".format(agg_sel[sel]['SHD_A_mean'], agg_sel[sel]['SHD_A_se']))
        print("  Mean Latent SHD for H: {:.4f} ± {:.4f}".format(agg_sel[sel]['vSHD_H_mean'], agg_sel[sel]['vSHD_H_se']))

    # If requested, print per replicate breakdowns for the SHD metrics.
    if verbose_graph:
        print_verbose_graphs_breakdown(replicates, edge_threshold=edge_threshold, v_threshold=v_eps)


def plot_replicate_diagnostics(results_data, true_A, true_H, exp_title=None, verbose=False):
    """
    For a single replicate, plot the evolution of MAE, correlation, and convergence metrics (if available)
    versus iterations. Vertical lines indicate the iterations where these metrics are minimized.
    Also print a summary of the key metrics.
    """
    num_iterations = len(results_data['est A values'])
    iterations = np.arange(1, num_iterations + 1)
    est_A_vals = results_data['est A values']
    est_H_vals = results_data['est H values']
    conv_scores = results_data['convergence scores']

    A_maes = [compute_mae(est_A_vals[i], true_A) for i in range(num_iterations)]
    H_maes = [compute_mae(est_H_vals[i], true_H) for i in range(num_iterations)]
    A_corrs = [calculate_correlation(est_A_vals[i], true_A) for i in range(num_iterations)]
    H_corrs = [calculate_correlation(est_H_vals[i], true_H) for i in range(num_iterations)]

    min_indices = get_min_indices(conv_scores)

    plot_MAEs(iterations, A_maes, H_maes, min_indices, exp_title=exp_title)
    # plot_correlation(iterations, A_corrs, H_corrs, min_indices, exp_title=exp_title)
    plot_normalized_convergence(results_data, exp_title=exp_title)

    if verbose:
        print("Iteration metrics summary:")
        print("Initial MAE (A):", A_maes[0])
        print("Final MAE (A):", A_maes[-1])
        print("Initial MAE (H):", H_maes[0])
        print("Final MAE (H):", H_maes[-1])
        print("Initial Corr (A):", A_corrs[0])
        print("Final Corr (A):", A_corrs[-1])
        print("Initial Corr (H):", H_corrs[0])
        print("Final Corr (H):", H_corrs[-1])
        for metric in ['nll', 'w2', 'w1', 'mmd']:
            if metric in min_indices:
                print(f"Min {metric.upper()} MAE (A) at iteration {min_indices[metric] + 1}:",
                      A_maes[min_indices[metric]])
                print(f"Min {metric.upper()} MAE (H) at iteration {min_indices[metric] + 1}:",
                      H_maes[min_indices[metric]])
                print(f"Min {metric.upper()} Corr (A) at iteration {min_indices[metric] + 1}:",
                      A_corrs[min_indices[metric]])
                print(f"Min {metric.upper()} Corr (H) at iteration {min_indices[metric] + 1}:",
                      H_corrs[min_indices[metric]])


def print_verbose_graphs_breakdown(replicates, edge_threshold=0.5, v_threshold=1):
    """
    For each replicate, print a breakdown of the SHD and v-SHD metrics
    for each selection method: WOT, APPEX, and the best iterations for available metrics.
    """
    print("\nPer replicate breakdown for SHD and v-SHD metrics:")
    for rep, rep_data in replicates.items():
        true_A = rep_data['true_A']
        true_H = rep_data['true_H']
        conv_scores = rep_data['convergence scores']
        num_iterations = len(rep_data['est A values'])
        print('True A:', np.round(true_A, decimals=2))
        print('True H:', np.round(true_H, decimals=2))


        # WOT: iteration 0
        A_wot = rep_data['est A values'][0]
        H_wot = rep_data['est H values'][0]
        shd_wot = compute_shd(A_wot, true_A, edge_threshold)
        vshd_wot = compute_v_structure_shd(true_H, H_wot, v_threshold)
        print(f"  WOT       - SHD_A: {shd_wot:.4f}, vSHD_H: {vshd_wot:.4f}")

        # APPEX: last iteration
        A_appex = rep_data['est A values'][-1]
        H_appex = rep_data['est H values'][-1]
        shd_appex = compute_shd(A_appex, true_A, edge_threshold)
        vshd_appex = compute_v_structure_shd(true_H, H_appex, v_threshold)
        print(f"  APPEX     - SHD_A: {shd_appex:.4f}, vSHD_H: {vshd_appex:.4f}")

        # For each available convergence metric, print its best iteration breakdown:
        if any('nll' in score for score in conv_scores):
            idx_nll = int(np.argmin([score.get('nll', np.inf) for score in conv_scores]))
            A_nll = rep_data['est A values'][idx_nll]
            H_nll = rep_data['est H values'][idx_nll]
            shd_nll = compute_shd(A_nll, true_A, edge_threshold)
            vshd_nll = compute_v_structure_shd(true_H, H_nll, v_threshold)
            print(f"  min NLL   - SHD_A: {shd_nll:.4f}, vSHD_H: {vshd_nll:.4f} (Iter {idx_nll + 1})")

        if any('w1' in score for score in conv_scores):
            idx_w1 = int(np.argmin([score.get('w1', np.inf) for score in conv_scores]))
            A_w1 = rep_data['est A values'][idx_w1]
            H_w1 = rep_data['est H values'][idx_w1]
            shd_w1 = compute_shd(A_w1, true_A, edge_threshold)
            vshd_w1 = compute_v_structure_shd(true_H, H_w1, v_threshold)
            print(f"  min W1    - SHD_A: {shd_w1:.4f}, vSHD_H: {vshd_w1:.4f} (Iter {idx_w1 + 1})")

        if any('w2' in score for score in conv_scores):
            idx_w2 = int(np.argmin([score.get('w2', np.inf) for score in conv_scores]))
            A_w2 = rep_data['est A values'][idx_w2]
            H_w2 = rep_data['est H values'][idx_w2]
            shd_w2 = compute_shd(A_w2, true_A, edge_threshold)
            vshd_w2 = compute_v_structure_shd(true_H, H_w2, v_threshold)
            print(f"  min W2    - SHD_A: {shd_w2:.4f}, vSHD_H: {vshd_w2:.4f} (Iter {idx_w2 + 1})")

        if any('mmd' in score for score in conv_scores):
            idx_mmd = int(np.argmin([score.get('mmd', np.inf) for score in conv_scores]))
            A_mmd = rep_data['est A values'][idx_mmd]
            H_mmd = rep_data['est H values'][idx_mmd]
            shd_mmd = compute_shd(A_mmd, true_A, edge_threshold)
            vshd_mmd = compute_v_structure_shd(true_H, H_mmd, v_threshold)
            print(f"  min MMD   - SHD_A: {shd_mmd:.4f}, vSHD_H: {vshd_mmd:.4f} (Iter {idx_mmd + 1})")


# ===============================
# Main Section
# ===============================
if __name__ == '__main__':
    # Example: causal sufficiency experiments
    # for d in [3]:
    #     for p in [0.1, 0.25]:
    #         directory = f"Results_experiment_causal_sufficiency_random_{d}_sparsity_{p}_seed-7"
    #         for eps in [0.45]:
    #             for v_eps in [0.9]:
    #                 interpret_experiment_directory(directory,
    #                                                exp_title=f"Causal Sufficiency Experiment (d={d}, p={p})",
    #                                                display_plots=True, v_eps=v_eps,
    #                                                edge_threshold=eps, causality_only=True, verbose_graph=True)

    for d in [3]:
        directory = f"Results_experiment_latent_confounder_random_{d}_sparsity_0.25_seed-0"
        for eps in [0.45]:
            for v_eps in [0.9]:
                # Call the interpretation function on the directory.
                interpret_experiment_directory(directory,
                                               exp_title=f"Latent Confounder Experiment (d={d}, p=0.25)",
                                               display_plots=True, v_eps=v_eps, edge_threshold=eps,
                                               causality_only=True, verbose_graph=True)
