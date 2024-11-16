import pickle
import math
import os
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def compute_shd(true_matrix, est_matrix, edge_threshold):
    '''
    Compute the Structural Hamming Distance (SHD) between two adjacency matrices, determined by drift A
    :param true_matrix: true drift matrix
    :param est_matrix: estimated drift matrix
    :param edge_threshold: threshold for determining presence of a simple edge
    :return: shd_signed: signed Structural Hamming Distance (SHD) between the true and estimated adjacency matrices
    '''
    d = true_matrix.shape[0]

    # Define sets for positive and negative edges in the true and estimated graphs
    true_edges_pos = set((i, j) for i in range(d) for j in range(d)
                         if true_matrix[j, i] > edge_threshold)
    true_edges_neg = set((i, j) for i in range(d) for j in range(d)
                         if true_matrix[j, i] < -edge_threshold)
    no_edges = set((i, j) for i in range(d) for j in range(d)) - (true_edges_pos | true_edges_neg)

    est_edges_pos = set((i, j) for i in range(d) for j in range(d)
                        if est_matrix[j, i] > edge_threshold)
    est_edges_neg = set((i, j) for i in range(d) for j in range(d)
                        if est_matrix[j, i] < -edge_threshold)
    no_edges_est = set((i, j) for i in range(d) for j in range(d)) - (est_edges_pos | est_edges_neg)

    # Final SHD includes all mismatches and discrepancies
    shd_signed = (len(no_edges.intersection(est_edges_pos)) + len(no_edges.intersection(est_edges_neg)) +
                  len(true_edges_neg.intersection(est_edges_pos)) + len(true_edges_neg.intersection(no_edges_est)) +
                  len(true_edges_pos.intersection(est_edges_neg))) + len(true_edges_pos.intersection(no_edges_est))

    return shd_signed

def compute_v_structure_shd(H_true, H_est, edge_threshold=0.5):
    '''
    Compute the Structural Hamming Distance (SHD) for v-structures between two adjacency matrices, determined by diffusion H
    :param H_true: true diffusion matrix
    :param H_est: estimated diffusion matrix
    :param edge_threshold: threshold for determining presence of a v-structure from a latent confounder
    :return: shd_v_structure: Structural Hamming Distance (SHD) for v-structures between the true and estimated adjacency matrices
    '''
    d = H_true.shape[0]
    shd_v_structure = 0

    # Loop over all pairs (i, j) where i != j
    for i in range(d):
        for j in range(d):
            if i != j:
                if abs(H_true[i, j]) > edge_threshold and abs(H_est[i, j]) <= edge_threshold:
                    shd_v_structure += 1
                elif abs(H_true[i, j]) <= edge_threshold and abs(H_est[i, j]) > edge_threshold:
                    shd_v_structure += 1
    return shd_v_structure / 2


def plot_causal_graphs(true_A, est_A_0, est_A_30, est_A_min_nll, true_H, est_H_0, est_H_30, est_H_min_nll,
                       edge_threshold=0.5, v_eps=1, display_plot=False, latent=True, min_nll_index=None):
    '''
    :param true_A:
    :param est_A_0:
    :param est_A_30:
    :param est_A_min_nll:
    :param true_H:
    :param est_H_0:
    :param est_H_30:
    :param est_H_min_nll:
    :param edge_threshold:
    :param v_eps:
    :param display_plot:
    :param latent:
    :param min_nll_index:
    :return:
    '''
    d = true_A.shape[0]
    # Calculate simple SHD for all estimated graphs
    shd_wot = compute_shd(true_A, est_A_0, edge_threshold)
    shd_appex = compute_shd(true_A, est_A_30, edge_threshold)
    shd_min_nll = compute_shd(true_A, est_A_min_nll, edge_threshold)
    # Calculate v-structure SHD for all estimated graphs
    v_shd_wot = compute_v_structure_shd(true_H, est_H_0, v_eps)
    v_shd_appex = compute_v_structure_shd(true_H, est_H_30, v_eps)
    v_shd_min_nll = compute_v_structure_shd(true_H, est_H_min_nll, v_eps)

    if display_plot:
        # Print SHD values
        print("Simple Structural Hamming Distance (SHD) between True Graph and Estimated Graph by WOT:", shd_wot)
        print("Simple Structural Hamming Distance (SHD) between True Graph and Estimated Graph by APPEX:", shd_appex)
        print("Simple Structural Hamming Distance (SHD) between True Graph and Estimated Graph by APPEX at min NLL:", shd_min_nll)
        print("V-structure SHD between True Graph and Estimated Graph by WOT:", v_shd_wot)
        print("V-structure SHD between True Graph and Estimated Graph by APPEX:", v_shd_appex)
        print("V-structure SHD between True Graph and Estimated Graph by APPEX at min NLL:", v_shd_min_nll)

        if latent:
            graphs = [
                (true_A, true_H, "True Causal Graph"),
                (est_A_0, est_H_0, "Estimated Causal Graph by WOT"),
                (est_A_30, est_H_30, "Estimated Causal Graph by APPEX"),
                (est_A_min_nll, est_H_min_nll, f"Estimated Causal Graph at Min NLL (Iteration {min_nll_index + 1})")
            ]

            # Initialize graphs for each scenario
            plot_graphs = [nx.DiGraph() for _ in range(len(graphs))]
            main_nodes = range(1, d + 1)
            pos = nx.circular_layout(list(main_nodes))

            # Helper function to position exogenous nodes
            def get_exogenous_position(pos, node, offset=0.3):
                x1, y1 = pos[node[0]]
                x2, y2 = pos[node[1]]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)
                dir_x, dir_y = dx / length, dy / length
                new_x = mid_x + dir_y * offset
                new_y = mid_y - dir_x * offset
                return (new_x, new_y)

            fig, axes = plt.subplots(1, len(graphs), figsize=(6 * len(graphs), 6))

            # Store min and max coordinates for axis limit adjustments
            all_x_values, all_y_values = [], []

            for idx, (A, H, title) in enumerate(graphs):
                graph = plot_graphs[idx]
                graph.add_nodes_from(main_nodes)

                # Plot A matrix connections
                for i in range(d):
                    for j in range(d):
                        if abs(A[j, i]) > edge_threshold:
                            color = 'red' if A[j, i] < 0 else 'green'
                            curved = graph.has_edge(i + 1, j + 1) or graph.has_edge(j + 1, i + 1)
                            graph.add_edge(i + 1, j + 1, color=color, curved=curved, style='solid')

                # Plot H matrix exogenous variable connections
                for i in range(d):
                    for j in range(i + 1, d):  # Only check upper triangular since H is symmetric
                        if abs(H[i, j]) >= v_eps:
                            exog_node = f'U_{i + 1}{j + 1}'
                            graph.add_node(exog_node)
                            pos[exog_node] = get_exogenous_position(pos, (i + 1, j + 1))
                            graph.add_edge(exog_node, i + 1, color='black', style='dotted')
                            graph.add_edge(exog_node, j + 1, color='black', style='dotted')

                # Collect coordinates for axis limits
                x_values, y_values = zip(*pos.values())
                all_x_values.extend(x_values)
                all_y_values.extend(y_values)

                # Plot the graph
                plot_single_graph(graph, pos, title, axes[idx])

            # Determine global axis limits
            xmin, xmax = min(all_x_values) - 0.5, max(all_x_values) + 0.5
            ymin, ymax = min(all_y_values) - 0.5, max(all_y_values) + 0.5

            for ax in axes:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            plt.tight_layout()
            plt.show()
        else:
            # Adjustments for latent=False
            graphs = [
                (true_A, "True Causal Graph"),
                (est_A_0, "Estimated Causal Graph by WOT"),
                (est_A_30, "Estimated Causal Graph by APPEX"),
                (est_A_min_nll, f"Estimated Causal Graph at Min NLL (Iteration {min_nll_index + 1})")
            ]

            # Initialize graphs for each scenario
            plot_graphs = [nx.DiGraph() for _ in range(len(graphs))]
            main_nodes = range(1, d + 1)
            pos = nx.circular_layout(list(main_nodes))

            fig, axes = plt.subplots(1, len(graphs), figsize=(6 * len(graphs), 6))

            # Store min and max coordinates for axis limit adjustments
            all_x_values, all_y_values = [], []

            for idx, (A, title) in enumerate(graphs):
                graph = plot_graphs[idx]
                graph.add_nodes_from(main_nodes)

                # Plot A matrix connections
                for i in range(d):
                    for j in range(d):
                        if abs(A[j, i]) > edge_threshold:
                            color = 'red' if A[j, i] < 0 else 'green'
                            curved = graph.has_edge(i + 1, j + 1) or graph.has_edge(j + 1, i + 1)
                            graph.add_edge(i + 1, j + 1, color=color, curved=curved, style='solid')

                # Collect coordinates for axis limits
                x_values, y_values = zip(*pos.values())
                all_x_values.extend(x_values)
                all_y_values.extend(y_values)

                # Plot the graph
                plot_single_graph(graph, pos, title, axes[idx])

            # Determine global axis limits
            xmin, xmax = min(all_x_values) - 0.5, max(all_x_values) + 0.5
            ymin, ymax = min(all_y_values) - 0.5, max(all_y_values) + 0.5

            for ax in axes:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            plt.tight_layout()
            plt.show()

    return shd_wot, shd_appex, shd_min_nll, v_shd_wot, v_shd_appex, v_shd_min_nll


def filter_redundant_edges(graph):
    filtered_edges = {}
    for u, v, data in graph.edges(data=True):
        key = (u, v)
        if key not in filtered_edges:
            filtered_edges[key] = data
        else:
            # Prioritize based on color, e.g., prioritize red over green
            if data.get('color') == 'red':
                filtered_edges[key] = data
            # If the existing edge is not red and the new one is, replace it
            elif filtered_edges[key].get('color') != 'red' and data.get('color') == 'red':
                filtered_edges[key] = data
    return filtered_edges

def plot_single_graph(graph, pos, title, ax):
    # Filter for unique edges to avoid conflicts
    filtered_edges = filter_redundant_edges(graph)

    # Separate edges by type (straight, curved, solid, and dotted)
    straight_edges = [(u, v) for (u, v), data in filtered_edges.items() if not data.get('curved', False)]
    curved_edges = [(u, v) for (u, v), data in filtered_edges.items() if data.get('curved', False)]
    dotted_edges = [(u, v) for (u, v), data in filtered_edges.items() if data.get('style') == 'dotted']
    solid_edges = [(u, v) for (u, v), data in filtered_edges.items() if data.get('style') == 'solid']

    # Assign edge colors
    edge_colors_straight = [filtered_edges[(u, v)]['color'] for u, v in straight_edges] if straight_edges else ['black']
    edge_colors_curved = [filtered_edges[(u, v)]['color'] for u, v in curved_edges] if curved_edges else ['black']

    # Handle self-loops by choosing one color (e.g., prioritize red self-loops)
    self_loops = [(u, v) for u, v in filtered_edges if u == v]
    if self_loops:
        self_loops = [(u, v) for u, v in self_loops if filtered_edges[(u, v)].get('color') == 'red']

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=800, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', ax=ax)

    # Draw solid straight edges
    if straight_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=straight_edges, edge_color=edge_colors_straight,
            arrows=True, arrowstyle='-|>', arrowsize=8, min_target_margin=15, ax=ax
        )

    # Draw curved edges with arc style
    if curved_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=curved_edges, edge_color=edge_colors_curved,
            arrows=True, arrowstyle='-|>', arrowsize=8, min_target_margin=15,
            connectionstyle='arc3,rad=0.3', ax=ax
        )

    # Draw dotted edges for exogenous nodes
    if dotted_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=dotted_edges, edge_color='black',
            arrows=True, arrowstyle='-|>', arrowsize=8, min_target_margin=15, ax=ax,
            style=(0, (15, 10))  # Custom dash pattern for long dashes
        )

    ax.set_aspect('equal')
    ax.set_title(title)

def aggregate_results(results_data_global, ground_truth_A_list, ground_truth_D_list, nll=False):
    """
    Aggregate estimated A and D values from experiment replicates and compute MAEs and correlations.
    Also compute the results for the iteration with the lowest average NLL across replicates.
    :param results_data_global: Dictionary with replicate keys containing estimated 'A' and 'D' values per iteration.
    :param ground_truth_A_list: List of ground truth 'A' matrices for each replicate.
    :param ground_truth_D_list: List of ground truth 'D' matrices for each replicate.
    :return: Aggregated MAEs, standard errors, correlations, and additional results for the minimal average NLL iteration.
    """
    num_iterations = 30  # Assuming there are 30 iterations

    # Lists to store the results per iteration
    A_mean_maes = []
    D_mean_maes = []
    A_mae_std_errs = []
    D_mae_std_errs = []

    # Lists to store correlations for each iteration
    A_correlations = []
    D_correlations = []
    A_cor_std_errs = []
    D_cor_std_errs = []

    # Initialize a list to store average NLL per iteration across replicates
    avg_nll_per_iteration = [0.0] * num_iterations

    num_replicates = len(results_data_global.keys())

    # First, compute MAEs, correlations, and NLLs per iteration across replicates
    for iteration in range(num_iterations):
        A_maes = []
        D_maes = []
        A_corrs = []
        D_corrs = []
        nlls_at_iteration = []

        # Loop through the experiment replicates
        for key in sorted(results_data_global.keys()):
            results_data = results_data_global[key]
            ground_truth_A = ground_truth_A_list[key - 1]
            ground_truth_D = ground_truth_D_list[key - 1]

            # Retrieve the estimated values for A and D at the current iteration
            A = results_data['est A values'][iteration]
            D = results_data['est D values'][iteration]

            if 'nll values' in results_data:
                # Retrieve the NLL value at the current iteration
                nll_value = results_data['nll values'][iteration]
                nll = True
            else:
                nll = False
            if nll:
                # Collect NLLs for averaging
                nlls_at_iteration.append(nll_value)

            # Compute MAE
            A_maes.append(compute_mae(A, ground_truth_A))
            D_maes.append(compute_mae(D, ground_truth_D))

            # Compute Correlations
            A_corr = calculate_correlation(A, ground_truth_A)
            D_corr = calculate_correlation(D, ground_truth_D)
            A_corrs.append(A_corr)
            D_corrs.append(D_corr)

        if nll:
            # Compute the average NLL for the current iteration across all replicates
            avg_nll = np.mean(nlls_at_iteration)
            avg_nll_per_iteration[iteration] = avg_nll

        # Compute the average MAE and correlations for the current iteration
        avg_A_mae = np.mean(A_maes)
        avg_D_mae = np.mean(D_maes)
        A_mean_maes.append(avg_A_mae)
        D_mean_maes.append(avg_D_mae)

        # Compute standard error of MAEs
        A_mae_std_errs.append(np.std(A_maes, ddof=1) / np.sqrt(num_replicates))
        D_mae_std_errs.append(np.std(D_maes, ddof=1) / np.sqrt(num_replicates))

        # Compute average correlations
        avg_A_corr = np.mean(A_corrs)
        avg_D_corr = np.mean(D_corrs)
        A_correlations.append(avg_A_corr)
        D_correlations.append(avg_D_corr)

        # Compute standard error of correlations
        A_cor_std_errs.append(np.std(A_corrs, ddof=1) / np.sqrt(num_replicates))
        D_cor_std_errs.append(np.std(D_corrs, ddof=1) / np.sqrt(num_replicates))
    if nll:
        # Find the iteration with minimal average NLL
        min_nll_index = np.argmin(avg_nll_per_iteration)

        # Now compute results for iteration with minimal average NLL
        A_maes_min_nll = []
        D_maes_min_nll = []
        A_corrs_min_nll = []
        D_corrs_min_nll = []

        # Loop through the experiment replicates
        for key in sorted(results_data_global.keys()):
            results_data = results_data_global[key]
            ground_truth_A = ground_truth_A_list[key - 1]
            ground_truth_D = ground_truth_D_list[key - 1]

            # Retrieve the estimated values for A and D at the iteration with minimal average NLL
            A = results_data['est A values'][min_nll_index]
            D = results_data['est D values'][min_nll_index]

            # Compute MAE
            A_maes_min_nll.append(compute_mae(A, ground_truth_A))
            D_maes_min_nll.append(compute_mae(D, ground_truth_D))

            # Compute Correlations
            A_corr = calculate_correlation(A, ground_truth_A)
            D_corr = calculate_correlation(D, ground_truth_D)
            A_corrs_min_nll.append(A_corr)
            D_corrs_min_nll.append(D_corr)

        # Compute average MAEs and correlations for the minimal average NLL iteration
        avg_A_mae_min_nll = np.mean(A_maes_min_nll)
        avg_D_mae_min_nll = np.mean(D_maes_min_nll)
        A_mae_min_nll_std_err = np.std(A_maes_min_nll, ddof=1) / np.sqrt(num_replicates)
        D_mae_min_nll_std_err = np.std(D_maes_min_nll, ddof=1) / np.sqrt(num_replicates)

        avg_A_corr_min_nll = np.mean(A_corrs_min_nll)
        avg_D_corr_min_nll = np.mean(D_corrs_min_nll)
        A_cor_min_nll_std_err = np.std(A_corrs_min_nll, ddof=1) / np.sqrt(num_replicates)
        D_cor_min_nll_std_err = np.std(D_corrs_min_nll, ddof=1) / np.sqrt(num_replicates)

    if nll:
        # Return the original outputs plus the new ones
        return (A_mean_maes, A_mae_std_errs, D_mean_maes, D_mae_std_errs,
                A_correlations, D_correlations, A_cor_std_errs, D_cor_std_errs,
                avg_A_mae_min_nll, A_mae_min_nll_std_err, avg_D_mae_min_nll, D_mae_min_nll_std_err,
                avg_A_corr_min_nll, avg_D_corr_min_nll, A_cor_min_nll_std_err, D_cor_min_nll_std_err,
                avg_nll_per_iteration, min_nll_index)
    else:
        return (A_mean_maes, A_mae_std_errs, D_mean_maes, D_mae_std_errs,
                A_correlations, D_correlations, A_cor_std_errs, D_cor_std_errs)


def compute_mae(estimated, ground_truth):
    """Compute Mean Absolute Percentage Error (MAE)"""
    mae = np.mean(np.abs((estimated - ground_truth)))
    return mae


def calculate_correlation(estimated_matrix, ground_truth_matrix):
    """Calculates the correlation between two matrices."""
    # Flatten the matrices to 1D arrays
    estimated_flat = estimated_matrix.flatten()
    ground_truth_flat = ground_truth_matrix.flatten()
    #  Calculate the Pearson correlation
    correlation = np.corrcoef(estimated_flat, ground_truth_flat)[0, 1]
    return correlation

def plot_mae_and_correlation_vs_iterations(results_data_version1, ground_truth_A1_list, ground_truth_GGT1_list,
                                           exp_title=None):
    # Aggregate the estimated A and GGT values from version 1
    if 'nll values' in results_data_version1[1]:
        (
            A_mean_maes_1, A_mae_std_errs_1, D_mean_maes_1, D_mae_std_errs_1,
            A_correlations_1, D_correlations_1, A_cor_std_errs_1, D_cor_std_errs_1,
            avg_A_mae_min_nll, A_mae_min_nll_std_err, avg_D_mae_min_nll, D_mae_min_nll_std_err,
            avg_A_corr_min_nll, avg_D_corr_min_nll, A_cor_min_nll_std_err, D_cor_min_nll_std_err,
            avg_nll_per_iteration, min_nll_index
        ) = aggregate_results(
            results_data_version1,
            ground_truth_A1_list,
            ground_truth_GGT1_list)
    else:
        (
            A_mean_maes_1, A_mae_std_errs_1, D_mean_maes_1, D_mae_std_errs_1,
            A_correlations_1, D_correlations_1, A_cor_std_errs_1, D_cor_std_errs_1
        ) = aggregate_results(
            results_data_version1,
            ground_truth_A1_list,
            ground_truth_GGT1_list)

    iterations = np.arange(1, len(A_mean_maes_1) + 1)

    # Plot the MAE for drift (A) and diffusion (GGT) with error bars
    plt.figure(figsize=(10, 6))

    # Plot MAE over iterations
    plt.errorbar(iterations, A_mean_maes_1, yerr=A_mae_std_errs_1,
                 label='MAE between estimated A and true A',
                 color='black', linestyle='-', marker='o')
    plt.errorbar(iterations, D_mean_maes_1, yerr=D_mae_std_errs_1,
                 label='MAE between estimated H and true H',
                 color='black', linestyle=':', marker='o',
                 markerfacecolor='none', markeredgecolor='black')

    # # Plot the MAE at minimal NLL iteration
    # plt.axvline(x=min_nll_index + 1, color='red', linestyle='--', label='Min NLL Iteration')
    # plt.scatter(min_nll_index + 1, avg_A_mae_min_nll, color='red', marker='x', s=100,
    #             label='MAE at Min NLL (A)')
    # plt.scatter(min_nll_index + 1, avg_D_mae_min_nll, color='red', marker='*', s=100,
    #             label='MAE at Min NLL (H)')

    # Print MAE values
    print('WOT MAE in A:', A_mean_maes_1[0], '+-', A_mae_std_errs_1[0])
    print('APPEX MAE in A:', A_mean_maes_1[-1], '+-', A_mae_std_errs_1[-1])
    if 'nll values' in results_data_version1[1]:
        print('Min NLL MAE in A:', avg_A_mae_min_nll, '+-', A_mae_min_nll_std_err)
    print('WOT MAE in H:', D_mean_maes_1[0], '+-', D_mae_std_errs_1[0])
    print('APPEX MAE in H:', D_mean_maes_1[-1], '+-', D_mae_std_errs_1[-1])
    if 'nll values' in results_data_version1[1]:
        print('Min NLL MAE in H:', avg_D_mae_min_nll, '+-', D_mae_min_nll_std_err)

    # Customize the MAE plot
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('MAE', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'MAE of Estimated Parameters vs Iterations for {exp_title}')
    else:
        plt.title(f'MAE of Estimated Parameters vs Iterations')
    plt.tight_layout()
    plt.show()

    # Create a second plot for correlations
    plt.figure(figsize=(10, 6))

    # Plot correlation over iterations
    plt.errorbar(iterations, A_correlations_1, yerr=A_cor_std_errs_1,
                 label='Correlation between estimated A and true A',
                 color='black', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='black')
    plt.errorbar(iterations, D_correlations_1, yerr=D_cor_std_errs_1,
                 label='Correlation between estimated H and true H',
                 color='black', linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')

    # # Plot the correlation at minimal NLL iteration
    # plt.axvline(x=min_nll_index + 1, color='red', linestyle='--', label='Min NLL Iteration')
    # plt.scatter(min_nll_index + 1, avg_A_corr_min_nll, color='red', marker='x', s=100,
    #             label='Correlation at Min NLL (A)')
    # plt.scatter(min_nll_index + 1, avg_D_corr_min_nll, color='red', marker='*', s=100,
    #             label='Correlation at Min NLL (H)')

    # Print correlation values
    print('WOT correlation in A:', A_correlations_1[0], '+-', A_cor_std_errs_1[0])
    print('APPEX correlation in A:', A_correlations_1[-1], '+-', A_cor_std_errs_1[-1])
    if 'nll values' in results_data_version1[1]:
        print('Min NLL correlation in A:', avg_A_corr_min_nll, '+-', A_cor_min_nll_std_err)
    print('WOT correlation in H:', D_correlations_1[0], '+-', D_cor_std_errs_1[0])
    print('APPEX correlation in H:', D_correlations_1[-1], '+-', D_cor_std_errs_1[-1])
    if 'nll values' in results_data_version1[1]:
        print('Min NLL correlation in H:', avg_D_corr_min_nll, '+-', D_cor_min_nll_std_err)

    # Customize the correlation plot
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Correlation', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'Correlation of Estimated Parameters vs Iterations for {exp_title}')
    else:
        plt.title(f'Correlation of Estimated Parameters vs Iterations')
    plt.tight_layout()
    plt.show()

    # Optionally, plot the average NLL over iterations
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, avg_nll_per_iteration, label='Average NLL', color='blue', marker='o')
    # plt.axvline(x=min_nll_index + 1, color='red', linestyle='--', label='Min NLL Iteration')
    # plt.xlabel('Iteration', fontsize=18)
    # plt.ylabel('Average NLL', fontsize=18)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # if exp_title is not None:
    #     plt.title(f'Average NLL over Iterations for {exp_title}')
    # else:
    #     plt.title(f'Average NLL over Iterations')
    # plt.tight_layout()
    # plt.show()



def retrieve_true_A_D(exp_number, version):
    if exp_number == 1:
        if version == 1:
            A = np.array([[-1]])
            G = np.eye(1)
        else:
            A = np.array([[-10]])
            G = math.sqrt(10) * np.eye(1)
    elif exp_number == 2:
        d = 2
        if version == 1:
            A = np.zeros((d, d))
        else:
            A = np.array([[0, 1], [-1, 0]])
        G = np.eye(d)
    elif exp_number == 3:
        d = 2
        if version == 1:
            A = np.array([[1, 2], [1, 0]])
        else:
            A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
        G = np.array([[1, 2], [-1, -2]])

    return A, np.matmul(G, G.T)


def compute_mse(estimated, ground_truth):
    """Compute Mean Squared Error (MSE)"""
    mse = np.mean((estimated - ground_truth) ** 2)
    return mse

def interpret_causal_experiment(directory_path, edge_threshold=0.5, v_eps=1, show_stats=False, display_plot=False,
                                latent=True):
    """
    This function processes replicate results to compute and display the Mean Squared Error (MSE) and
    Structural Hamming Distance (SHD) between estimated and true A/D matrices at:
    - Initial iteration (WOT)
    - Iteration 30 (APPEX)
    - Iteration with minimal NLL (APPEX with min NLL)

    Parameters:
    - directory_path: Path to the directory containing the replicate pickle files.
    - edge_threshold: Threshold for determining edges in the estimated matrices (for A matrices).
    - v_eps: Threshold for determining v-structures in the estimated matrices (for D matrices).
    - show_stats: If True, prints detailed statistics.
    - display_plot: If True, displays plots (functionality needs to be implemented).
    - latent: If True, computes v-structure SHD.
    """

    N_values = []
    A_mse_values_30 = []
    D_mse_values_30 = []
    A_mse_values_min_nll = []
    D_mse_values_min_nll = []

    # Lists to store filenames and extracted N values
    files_with_N = []

    # Iterate over all .pkl files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            # Extract N from the filename using regex
            match = re.search(r'_N-(\d+)', filename)
            if match:
                N = int(match.group(1))
                files_with_N.append((N, filename))

    # Sort files based on extracted N values
    files_with_N.sort()
    shd_wot_list = []
    shd_appex_list = []
    shd_min_nll_list = []
    if latent:
        v_shd_wot_list = []
        v_shd_appex_list = []
        v_shd_min_nll_list = []

    # Process files in order of N
    for N, filename in files_with_N:
        file_path = os.path.join(directory_path, filename)

        # Load the replicate data
        with open(file_path, 'rb') as f:
            results_data = pickle.load(f)
        print(file_path)
        # Extract true A, true D, est A values, est D values
        true_A = results_data['true_A']
        true_D = results_data['true_D']
        est_A_values = results_data['est A values']
        est_D_values = results_data['est D values']
        nll_values = results_data['nll values']

        # Get the estimated A and D at iteration 1 (index 0) and iteration 30 (index 29)
        est_A_0 = est_A_values[0]
        est_A_30 = est_A_values[29]
        est_D_0 = est_D_values[0]
        est_D_30 = est_D_values[29]

        # Find the iteration with minimal NLL
        min_nll_index = np.argmin(nll_values)
        est_A_min_nll = est_A_values[min_nll_index]
        est_D_min_nll = est_D_values[min_nll_index]

        if show_stats:

            print('True D:', true_D)
            print('Initial D:', results_data['initial D'])
            print('True A:', true_A)
            print('Estimated A by WOT (Iteration 1):', est_A_0)
            print('Estimated A after 30 iterations:', est_A_30)
            print(f'Estimated A at min NLL (Iteration {min_nll_index + 1}):', est_A_min_nll)
            print('Estimated D by WOT (Iteration 1):', est_D_0)
            print('Estimated D after 30 iterations:', est_D_30)
            print(f'Estimated D at min NLL (Iteration {min_nll_index + 1}):', est_D_min_nll)
            plot_results_for_replicate(results_data, true_A, true_D)

        if edge_threshold is not None:
            # Compute SHD and plot causal graphs
            shd_wot, shd_appex, shd_min_nll, v_shd_wot, v_shd_appex, v_shd_min_nll = plot_causal_graphs(
                true_A, est_A_0, est_A_30, est_A_min_nll,
                true_D, est_D_0, est_D_30, est_D_min_nll,
                edge_threshold=edge_threshold,
                v_eps=v_eps,
                display_plot=display_plot, latent=latent, min_nll_index=min_nll_index)
            shd_wot_list.append(shd_wot)
            shd_appex_list.append(shd_appex)
            shd_min_nll_list.append(shd_min_nll)
            if latent:
                v_shd_wot_list.append(v_shd_wot)
                v_shd_appex_list.append(v_shd_appex)
                v_shd_min_nll_list.append(v_shd_min_nll)

        # Compute the MSE for A and D at iteration 30 and at min NLL iteration
        A_mse_30 = compute_mse(est_A_30, true_A)
        D_mse_30 = compute_mse(est_D_30, true_D)
        A_mse_min_nll = compute_mse(est_A_min_nll, true_A)
        D_mse_min_nll = compute_mse(est_D_min_nll, true_D)

        # Append the results
        N_values.append(N)
        A_mse_values_30.append(A_mse_30)
        D_mse_values_30.append(D_mse_30)
        A_mse_values_min_nll.append(A_mse_min_nll)
        D_mse_values_min_nll.append(D_mse_min_nll)

    # Calculate mean and standard error for SHD WOT
    mean_shd_wot = np.mean(shd_wot_list)
    std_error_shd_wot = np.std(shd_wot_list, ddof=1) / np.sqrt(len(shd_wot_list))

    # Calculate mean and standard error for SHD APPEX
    mean_shd_appex = np.mean(shd_appex_list)
    std_error_shd_appex = np.std(shd_appex_list, ddof=1) / np.sqrt(len(shd_appex_list))

    # Calculate mean and standard error for SHD APPEX at min NLL
    mean_shd_min_nll = np.mean(shd_min_nll_list)
    std_error_shd_min_nll = np.std(shd_min_nll_list, ddof=1) / np.sqrt(len(shd_min_nll_list))

    # Print the results
    print("\nSimple SHD Results:")
    print("Mean SHD WOT:", mean_shd_wot, "±", std_error_shd_wot)
    print("Mean SHD APPEX:", mean_shd_appex, "±", std_error_shd_appex)
    print("Mean SHD APPEX at min NLL:", mean_shd_min_nll, "±", std_error_shd_min_nll)

    if latent:
        # Calculate mean and standard error for v-structure SHD WOT
        v_mean_shd_wot = np.mean(v_shd_wot_list)
        v_std_error_shd_wot = np.std(v_shd_wot_list, ddof=1) / np.sqrt(len(v_shd_wot_list))

        # Calculate mean and standard error for v-structure SHD APPEX
        v_mean_shd_appex = np.mean(v_shd_appex_list)
        v_std_error_shd_appex = np.std(v_shd_appex_list, ddof=1) / np.sqrt(len(v_shd_appex_list))

        # Calculate mean and standard error for v-structure SHD APPEX at min NLL
        v_mean_shd_min_nll = np.mean(v_shd_min_nll_list)
        v_std_error_shd_min_nll = np.std(v_shd_min_nll_list, ddof=1) / np.sqrt(len(v_shd_min_nll_list))

        # Print the results
        print("\nV-Structure SHD Results:")
        print("Mean v-structure SHD WOT:", v_mean_shd_wot, "±", v_std_error_shd_wot)
        print("Mean v-structure SHD APPEX:", v_mean_shd_appex, "±", v_std_error_shd_appex)
        print("Mean v-structure SHD APPEX at min NLL:", v_mean_shd_min_nll, "±", v_std_error_shd_min_nll)


def plot_results_for_replicate(results_data, ground_truth_A, ground_truth_D, exp_title=None):
    """
    Plots the MAE, correlation, and NLL over iterations for a single replicate.
    Highlights the iteration with the minimal NLL.

    :param results_data: Dictionary containing the results for a single replicate.
    :param ground_truth_A: Ground truth A matrix for the replicate.
    :param ground_truth_D: Ground truth D matrix for the replicate.
    :param exp_title: Optional title for the plots.
    """
    num_iterations = len(results_data['est A values'])
    iterations = np.arange(1, num_iterations + 1)

    # Retrieve estimated A, D, and NLL values over iterations
    est_A_values = results_data['est A values']
    est_D_values = results_data['est D values']
    nll_values = results_data['nll values']

    # Initialize lists to store MAEs and correlations
    A_maes = []
    D_maes = []
    A_corrs = []
    D_corrs = []

    # Compute MAEs and correlations at each iteration
    for iteration in range(num_iterations):
        A_est = est_A_values[iteration]
        D_est = est_D_values[iteration]

        # Compute MAE
        A_mae = compute_mae(A_est, ground_truth_A)
        D_mae = compute_mae(D_est, ground_truth_D)
        A_maes.append(A_mae)
        D_maes.append(D_mae)

        # Compute correlation
        A_corr = calculate_correlation(A_est, ground_truth_A)
        D_corr = calculate_correlation(D_est, ground_truth_D)
        A_corrs.append(A_corr)
        D_corrs.append(D_corr)

    # Find the iteration with minimal NLL
    min_nll_index = np.argmin(nll_values)

    # Plot MAE over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, A_maes, label='MAE between estimated A and true A',
             color='black', linestyle='-', marker='o')
    plt.plot(iterations, D_maes, label='MAE between estimated H and true H',
             color='black', linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')

    # Highlight the minimal NLL iteration
    plt.axvline(x=min_nll_index + 1, color='red', linestyle='--', label='Min NLL Iteration')
    plt.scatter(min_nll_index + 1, A_maes[min_nll_index], color='red', marker='x', s=100,
                label='MAE at Min NLL (A)')
    plt.scatter(min_nll_index + 1, D_maes[min_nll_index], color='red', marker='*', s=100,
                label='MAE at Min NLL (H)')

    # Customize the MAE plot
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('MAE', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'MAE of Estimated Parameters vs Iterations for {exp_title}')
    else:
        plt.title(f'MAE of Estimated Parameters vs Iterations')
    plt.tight_layout()
    plt.show()

    # Plot correlation over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, A_corrs, label='Correlation between estimated A and true A',
             color='black', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='black')
    plt.plot(iterations, D_corrs, label='Correlation between estimated H and true H',
             color='black', linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')

    # Highlight the minimal NLL iteration
    plt.axvline(x=min_nll_index + 1, color='red', linestyle='--', label='Min NLL Iteration')
    plt.scatter(min_nll_index + 1, A_corrs[min_nll_index], color='red', marker='x', s=100,
                label='Correlation at Min NLL (A)')
    plt.scatter(min_nll_index + 1, D_corrs[min_nll_index], color='red', marker='*', s=100,
                label='Correlation at Min NLL (H)')

    # Customize the correlation plot
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Correlation', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'Correlation of Estimated Parameters vs Iterations for {exp_title}')
    else:
        plt.title(f'Correlation of Estimated Parameters vs Iterations')
    plt.tight_layout()
    plt.show()

    # Plot NLL over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, nll_values, label='NLL', color='blue', marker='o')
    plt.axvline(x=min_nll_index + 1, color='red', linestyle='--', label='Min NLL Iteration')
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('NLL', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'NLL over Iterations for {exp_title}')
    else:
        plt.title(f'NLL over Iterations')
    plt.tight_layout()
    plt.show()

    # Print MAE and correlation at initial, final, and min NLL iterations
    print('WOT MAE in A:', A_maes[0])
    print('APPEX (iteration 30) MAE in A:', A_maes[-1])
    print(f'APPEX (Min NLL) MAE in A at iteration {min_nll_index}:', A_maes[min_nll_index])
    print('WOT MAE in H:', D_maes[0])
    print('APPEX (iteration 30) MAE in H:', D_maes[-1])
    print(f'APPEX (Min NLL) MAE in D at iteration {min_nll_index}:', D_maes[min_nll_index])
    print('WOT Correlation in A:', A_corrs[0])
    print('APPEX (iteration 30) Correlation in A:', A_corrs[-1])
    print(f'APPEX (Min NLL)  Correlation in A:', A_corrs[min_nll_index])
    print('WOT Correlation in H:', D_corrs[0])
    print('APPEX (iteration 30) correlation in H:', D_corrs[-1])
    print(f'APPEX (Min NLL) Correlation in H:', D_corrs[min_nll_index])



def plot_exp_results(exp_number, version=None, d=None, num_reps=2, N=500, seed=1, plot_individual_results=True):
    print('Dimension:', d)
    results_data_global = {}
    ground_truth_A_list = []
    ground_truth_D_list = []
    for i in range(1, num_reps + 1):
        if exp_number != "random":
            filename = f'Results_experiment_{exp_number}_seed-{seed}/version-{version}_N-{N}_replicate-{i}.pkl'
        else:
            filename = f'Results_experiment_{exp_number}_{d}_seed-{seed}/replicate-{i}_N-{N}.pkl'
        with open(filename, 'rb') as f:
            results_data = pickle.load(f)
        if exp_number == 'random':
            ground_truth_A_list.append(results_data['true_A'])
            ground_truth_D_list.append(results_data['true_D'])

        results_data_global[i] = results_data
        if plot_individual_results:
            plot_results_for_replicate(results_data, results_data['true_A'], results_data['true_D'])


    if exp_number != 'random':
        ground_truth_A1, ground_truth_GGT1 = retrieve_true_A_D(exp_number, version)
        ground_truth_A_list = [ground_truth_A1] * num_reps
        ground_truth_D_list = [ground_truth_GGT1] * num_reps

    if exp_number == 'random':
        plot_mae_and_correlation_vs_iterations(results_data_global, ground_truth_A_list, ground_truth_D_list,
                                               exp_title=f'random SDEs of dimension {d}')
    else:
        plot_mae_and_correlation_vs_iterations(results_data_global, ground_truth_A_list, ground_truth_D_list,
                                               exp_title=f'SDE {version} from example {exp_number}')


'''
# Example usage
'''
# plot_exp_results(exp_number = 2, version = 1, num_reps=10, seed=1)
# plot_exp_results(exp_number = 2, version = 2, num_reps=10, seed=1)
# plot_exp_results(exp_number = 3, version = 1, num_reps=10, seed=1)
# plot_exp_results(exp_number = 3, version = 2, num_reps=10, seed=1)
# plot_exp_results(exp_number = 1, version = 1, num_reps=10)
# plot_exp_results(exp_number='random', d=3, num_reps=10, seed=69)
# plot_exp_results(exp_number='random', d=4, num_reps=10, seed=69)
# plot_exp_results(exp_number='random', d=5, num_reps=10, seed=69)
# plot_exp_results(exp_number='random', d=10, num_reps=10, seed=42)
# plot_exp_results(exp_number='random', d=3, num_reps=1, seed=9)
ds = [5]
ps = [0.25]
seeds = [69]
# seeds = np.arange(3,19)
for d in ds:
    for p in ps:
        for seed in seeds:
            # directory_path = f'Results_experiment_causal_sufficiency_random_{d}_sparsity_{p}_seed-{seed}'
            directory_path = f'Results_experiment_latent_confounder_random_{d}_sparsity_{p}_seed-69'
            interpret_causal_experiment(directory_path, show_stats=True, display_plot=True, latent=True, edge_threshold=0.5,
                                       v_eps=0.5)
