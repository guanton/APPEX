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

def plot_causal_graphs(true_A, est_A_0, est_A_30, true_H, est_H_0, est_H_30, edge_threshold=0.5, v_eps=1,
                              display_plot=False, latent=True):
    d = true_A.shape[0]
    # Calculate simple SHD for both estimated graphs
    shd_wot = compute_shd(true_A, est_A_0, edge_threshold)
    shd_appex = compute_shd(true_A, est_A_30, edge_threshold)
    # Calculate v-structure SHD for both estimated graphs
    v_shd_wot = compute_v_structure_shd(true_H, est_H_0, v_eps)
    v_shd_appex = compute_v_structure_shd(true_H, est_H_30, v_eps)
    if display_plot:
        # Compute Structural Hamming Distance (SHD)
        print("Simple Structural Hamming Distance (SHD) between True Graph and Estimated Graph by WOT:", shd_wot)
        print("Simple Structural Hamming Distance (SHD) between True Graph and Estimated Graph by APPEX:", shd_appex)
        # Compute Structural Hamming Distance (SHD)
        print("V-structure Structural Hamming Distance (SHD) between True Graph and Estimated Graph by WOT:", v_shd_wot)
        print("V-structure Structural Hamming Distance (SHD) between True Graph and Estimated Graph by APPEX:",
              v_shd_appex)


    if latent:
        graphs = [
            (true_A, true_H, "True Causal Graph"),
            (est_A_0, est_H_0, "Estimated Causal Graph by WOT"),
            (est_A_30, est_H_30, "Estimated Causal Graph by APPEX")
        ]
    
        # Initialize graphs for each scenario
        plot_graphs = [nx.DiGraph() for _ in range(3)]
        # Define layout positions
        main_nodes = range(1, d + 1)
        pos = nx.circular_layout(list(main_nodes))

    
        # Adjust positions for exogenous variables by extending the layout
        def get_exogenous_position(pos, node, offset=0.3):
            x1, y1 = pos[node[0]]
            x2, y2 = pos[node[1]]
    
            # Midpoint between (x1, y1) and (x2, y2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
    
            # Calculate direction vector (x2 - x1, y2 - y1) and normalize it
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)
            dir_x, dir_y = dx / length, dy / length
    
            # Offset the midpoint slightly away from the centerline for clarity
            new_x = mid_x + dir_y * offset
            new_y = mid_y - dir_x * offset
    
            return (new_x, new_y)
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
                        # Check if there is already an edge between (i+1, j+1) or (j+1, i+1)
                        if graph.has_edge(i + 1, j + 1) or graph.has_edge(j + 1, i + 1):
                            curved = True  # Apply curvature for parallel edges
                        else:
                            curved = abs(A[i, j]) > edge_threshold
                        graph.add_edge(i + 1, j + 1, color=color, curved=curved, style='solid')
    
            # Plot H matrix exogenous variable connections
            for i in range(d):
                for j in range(i + 1, d):  # Only check upper triangular since H is symmetric
                    if abs(H[i, j]) >= 0.5:
                        exog_node = f'U_{i + 1}{j + 1}'
                        graph.add_node(exog_node)
                        pos[exog_node] = get_exogenous_position(pos, (i + 1, j + 1))
                        # Assign dashed, black edges only for exogenous connections
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
    else:
        d = true_A.shape[0]
        true_graph = nx.DiGraph()
        est_graph_0 = nx.DiGraph()
        est_graph_30 = nx.DiGraph()

        # Add nodes (starting from 1 instead of 0)
        nodes = range(1, d + 1)
        true_graph.add_nodes_from(nodes)
        est_graph_0.add_nodes_from(nodes)
        est_graph_30.add_nodes_from(nodes)

        # Add edges with color coding and curvature for bidirectional connections
        graphs = [(true_A, true_graph), (est_A_0, est_graph_0), (est_A_30, est_graph_30)]
        for A, graph in graphs:
            for i in range(d):
                for j in range(d):
                    if abs(A[j, i]) > edge_threshold:
                        color = 'red' if A[j, i] < 0 else 'green'
                        # Check if it's a self-loop
                        if i == j:
                            graph.add_edge(i + 1, i + 1, color=color, curved=True)  # No curvature needed for self-loops
                        else:
                            if abs(A[i, j]) > edge_threshold:
                                graph.add_edge(i + 1, j + 1, color=color, curved=True)
                            else:
                                graph.add_edge(i + 1, j + 1, color=color, curved=False)

        # Set up the figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pos = nx.circular_layout(true_graph)

        plot_single_graph(true_graph, pos, 'True Causal Graph', axes[0])
        plot_single_graph(est_graph_0, pos, 'Estimated Causal Graph by WOT', axes[1])
        plot_single_graph(est_graph_30, pos, 'Estimated Causal Graph by APPEX', axes[2])

        # Dynamic axis limits
        x_values, y_values = zip(*pos.values())
        xmin, xmax = min(x_values) - 0.5, max(x_values) + 0.5
        ymin, ymax = min(y_values) - 0.5, max(y_values) + 0.5
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        plt.tight_layout()

    if display_plot:
        plt.show()
    return shd_wot, shd_appex, v_shd_wot, v_shd_appex

def plot_single_graph(graph, pos, title, ax):
    # Separate edges by type (straight, curved, solid, and dotted)
    straight_edges = [(u, v) for u, v, curved in graph.edges(data='curved') if not curved]
    curved_edges = [(u, v) for u, v, curved in graph.edges(data='curved') if curved]
    dotted_edges = [(u, v) for u, v, style in graph.edges(data='style') if style == 'dotted']
    solid_edges = [(u, v) for u, v, style in graph.edges(data='style') if style == 'solid']

    # Combine edge types and assign their respective colors
    edge_colors_straight = [graph[u][v]['color'] for u, v in straight_edges]
    edge_colors_curved = [graph[u][v]['color'] for u, v in curved_edges]

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=800, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', ax=ax)

    # Draw solid straight edges
    nx.draw_networkx_edges(
        graph, pos, edgelist=solid_edges, edge_color=edge_colors_straight,
        arrows=True, arrowstyle='-|>', arrowsize=8, min_target_margin=15, ax=ax
    )

    # Draw curved edges with arc style
    nx.draw_networkx_edges(
        graph, pos, edgelist=curved_edges, edge_color=edge_colors_curved,
        arrows=True, arrowstyle='-|>', arrowsize=8, min_target_margin=15,
        connectionstyle='arc3,rad=0.3', ax=ax
    )

    # Draw dotted edges for exogenous nodes
    nx.draw_networkx_edges(
        graph, pos, edgelist=dotted_edges, edge_color='black',
        arrows=True, arrowstyle='-|>', arrowsize=8, min_target_margin=15, ax=ax,
        style=(0, (15, 10))  # Custom dash pattern for long dashes
    )

    ax.set_aspect('equal')
    ax.set_title(title)


def aggregate_results(results_data, ground_truth_A_list, ground_truth_D_list):
    """Aggregate estimated A and GGT values from a dictionary where the keys are iteration numbers.
       Averages results for each iteration across 10 experiment iterates and computes correlations."""
    num_iterations = 30

    # Lists to store the results
    A_mean_maes = []
    D_mean_maes = []
    A_mae_std_errs = []
    D_mae_std_errs = []

    # Lists to store correlations for each iteration
    A_correlations = []
    D_correlations = []

    for iteration in range(num_iterations):
        A_maes = []
        D_maes = []
        A_corrs = []
        D_corrs = []

        # Loop through the experiment replicates
        for key in sorted(results_data.keys()):
            ground_truth_A = ground_truth_A_list[key - 1]
            ground_truth_D = ground_truth_D_list[key - 1]

            # Retrieve the estimated values for A and D at the current iteration
            A = results_data[key]['est A values'][iteration]
            D = results_data[key]['est D values'][iteration]

            # Compute MAE
            A_maes.append(compute_mae(A, ground_truth_A))
            D_maes.append(compute_mae(D, ground_truth_D))

            # Compute Correlations
            A_corr = calculate_correlation(A, ground_truth_A)
            D_corr = calculate_correlation(D, ground_truth_D)
            A_corrs.append(A_corr)
            D_corrs.append(D_corr)

        # Compute the average MAE and correlations for the current iteration
        avg_A_mae = np.mean(A_maes)
        avg_D_mae = np.mean(D_maes)
        A_mean_maes.append(avg_A_mae)
        D_mean_maes.append(avg_D_mae)

        # Compute standard error
        A_mae_std_errs.append(np.std(A_maes) / np.sqrt(len(results_data.keys())))
        D_mae_std_errs.append(np.std(D_maes) / np.sqrt(len(results_data.keys())))

        # Compute average correlations
        avg_A_corr = np.mean(A_corrs)
        avg_D_corr = np.mean(D_corrs)
        A_correlations.append(avg_A_corr)
        D_correlations.append(avg_D_corr)

    return A_mean_maes, A_mae_std_errs, D_mean_maes, D_mae_std_errs, A_correlations, D_correlations


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
    A_mean_maes_1, A_mae_std_errs_1, D_mean_maes_1, D_mae_std_errs_1, A_correlations_1, D_correlations_1 = aggregate_results(
        results_data_version1,
        ground_truth_A1_list,
        ground_truth_GGT1_list)

    iterations = np.arange(1, len(A_mean_maes_1) + 1)

    # Plot the MAE for drift (A) and diffusion (GGT) for both versions with error bars
    plt.figure(figsize=(10, 6))

    # Version 1 (with error bars)
    plt.errorbar(iterations, A_mean_maes_1, yerr=A_mae_std_errs_1, label='MAE between estimated A and true A',
                 color='black',
                 linestyle='-', marker='o')
    plt.errorbar(iterations, D_mean_maes_1, yerr=D_mae_std_errs_1, label='MAE between estimated H and true H',
                 color='black',
                 linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')

    # Customize the MAE plot
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'MAE of Estimated Parameters vs Iterations for {exp_title}')
    else:
        plt.title(f'MAE of Estimated Parameters vs Iterations')
    plt.show()

    # Create a second plot for correlations
    plt.figure(figsize=(10, 6))

    # Plot correlation for version 1
    plt.errorbar(iterations, A_correlations_1, label='Correlation between estimated A and true A', color='black',
                 linestyle='-', marker='o')
    plt.errorbar(iterations, D_correlations_1, label='Correlation between estimated H and true H', color='black',
                 linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='black')

    # Customize the correlation plot
    plt.xlabel('Iteration')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    if exp_title is not None:
        plt.title(f'Correlation of Estimated Parameters vs Iterations for {exp_title}')
    else:
        plt.title(f'Correlation of Estimated Parameters vs Iterations')
    plt.show()


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


def plot_exp_results(exp_number, version=None, d=None, num_reps=10):
    results_data_global = {}
    ground_truth_A_list = []
    ground_truth_D_list = []
    for i in range(1, num_reps + 1):
        if exp_number != "random":
            filename = f'Results_experiment_{exp_number}/version-{version}_replicate-{i}.pkl'
        else:
            filename = f'Results_experiment_{exp_number}_{d}/replicate-{i}.pkl'
        with open(filename, 'rb') as f:
            results_data = pickle.load(f)
        if exp_number == 'random':
            ground_truth_A_list.append(results_data['true_A'])
            ground_truth_D_list.append(results_data['true_D'])

        results_data_global[i] = results_data

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


def compute_mse(estimated, ground_truth):
    """Compute Mean Squared Error (MSE)"""
    mse = np.mean((estimated - ground_truth) ** 2)
    return mse


def interpret_causal_experiment(directory_path, edge_threshold=0.5, v_eps=1, show_stats=False, display_plot=False,
                               latent=True):
    """
    This function plots the MSE between estimated and true A/D at iteration 30 versus the value of N.

    Parameters:
    - directory_path: Path to the directory containing the replicate pickle files.
    """

    N_values = []
    A_mse_values = []
    D_mse_values = []

    # List to store filenames and extracted N values
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
    if latent:
        v_shd_wot_list = []
        v_shd_appex_list = []
    # Process files in order of N
    for N, filename in files_with_N:
        file_path = os.path.join(directory_path, filename)

        # Load the replicate data
        with open(file_path, 'rb') as f:
            results_data = pickle.load(f)
        print(file_path)
        # Extract true A, true D, est A values, est D values
        true_A = results_data['true_A']
        eigenvalues = np.linalg.eigvals(true_A)
        max_eigenvalue = float(np.max(eigenvalues))
        # print('Spectrum:', eigenvalues)
        # print('max eigenvalue:', max_eigenvalue)
        true_D = results_data['true_D']
        X0 = results_data['X0 points']
        est_A_values = results_data['est A values']
        est_D_values = results_data['est D values']
        true_G = results_data['true_G']

        # Get the estimated A and D at iteration 30 (index 29)
        est_A_0 = est_A_values[0]
        est_A_30 = est_A_values[29]
        est_D_0 = est_D_values[0]
        est_D_30 = est_D_values[29]

        if show_stats:
            print('True G:', true_G)
            print('True D:', true_D)
            print('Initial D:', results_data['initial D'])
            print('true A:', true_A)
            print('estimated A by WOT:', est_A_0)
            print('estimated A after 30:', est_A_30)

            print('estimated D by WOT:', est_D_0)
            print('estimated D after 30:', est_D_30)

        if edge_threshold is not None:
            shd_wot, shd_appex, v_shd_wot, v_shd_appex = plot_causal_graphs(true_A, est_A_0, est_A_30,
                                                                                   true_D, est_D_0, est_D_30,
                                                                                   edge_threshold=edge_threshold,
                                                                                   v_eps=v_eps,
                                                                                   display_plot=display_plot, latent=latent)
            shd_wot_list.append(shd_wot)
            shd_appex_list.append(shd_appex)
            # print('SHD WOT:', shd_wot)
            # print('SHD APPEX:', shd_appex)
            # plot_causal_graphs_latent(true_A, est_A_0, est_A_30, true_D, est_D_0, est_D_30, edge_threshold=0.5)


        # Compute the MSE for A and D at iteration 30
        A_mse = compute_mse(est_A_30, true_A)
        D_mse = compute_mse(est_D_30, true_D)

        # Append the results
        N_values.append(N)
        A_mse_values.append(A_mse)
        D_mse_values.append(D_mse)

    # Calculate mean and standard error for SHD WOT
    mean_shd_wot = np.mean(shd_wot_list)
    std_error_shd_wot = np.std(shd_wot_list, ddof=1) / np.sqrt(len(shd_wot_list))

    # Calculate mean and standard error for SHD APPEX
    mean_shd_appex = np.mean(shd_appex_list)
    std_error_shd_appex = np.std(shd_appex_list, ddof=1) / np.sqrt(len(shd_appex_list))

    # Print the results
    print("Mean simple SHD WOT:", mean_shd_wot)
    print("Standard Error simple SHD WOT:", std_error_shd_wot)
    print("Mean simple SHD APPEX:", mean_shd_appex)
    print("Standard Error simple SHD APPEX:", std_error_shd_appex)
    if latent:
        # Calculate mean and standard error for SHD WOT
        v_mean_shd_wot = np.mean(v_shd_wot_list)
        v_std_error_shd_wot = np.std(v_shd_wot_list, ddof=1) / np.sqrt(len(v_shd_wot_list))

        # Calculate mean and standard error for SHD APPEX
        v_mean_shd_appex = np.mean(v_shd_appex_list)
        v_std_error_shd_appex = np.std(v_shd_appex_list, ddof=1) / np.sqrt(len(v_shd_appex_list))

        # Print the results
        print("Mean v-structure SHD WOT:", v_mean_shd_wot)
        print("Standard Error v-structure SHD WOT:", v_std_error_shd_wot)
        print("Mean v-structure SHD APPEX:", v_mean_shd_appex)
        print("Standard Error v-structure SHD APPEX:", v_std_error_shd_appex)


def plot_exp_results(exp_number, version=None, d=None, num_reps=2, N=500, seed=0):
    results_data_global = {}
    ground_truth_A_list = []
    ground_truth_D_list = []
    for i in range(1, num_reps + 1):
        if exp_number != "random":
            filename = f'Results_experiment_{exp_number}_seed-{seed}/version-{version}_N-{N}_replicate-{i}.pkl'
        else:
            filename = f'Results_experiment_{exp_number}_{d}/replicate-{i}.pkl'
        with open(filename, 'rb') as f:
            results_data = pickle.load(f)
        if exp_number == 'random':
            ground_truth_A_list.append(results_data['true_A'])
            ground_truth_D_list.append(results_data['true_D'])

        results_data_global[i] = results_data

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


# plot_exp_results(exp_number='random', d=50, num_reps=10)
# plot_exp_results(exp_number = 1, version = 1, num_reps=2, seed=1)
# plot_exp_results(exp_number = 2, version = 2, num_reps=10)

ds = [10]
ps = [0.1]

for d in ds:
    for p in ps:
        directory_path = f'Results_experiment_causal_sufficiency_random_{d}_sparsity_{p}_seed-1'
        # directory_path = f'Results_experiment_latent_confounder_random_{d}_sparsity_{p}_seed-1'
        interpret_causal_experiment(directory_path, show_stats=True, display_plot=True, latent=True, edge_threshold=0.5,
                                   v_eps=0.5)
