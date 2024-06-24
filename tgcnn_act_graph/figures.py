import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np

from tgcnn_act_graph import utils, max_act_diff, calculations

def plot_activation_difference(filter_num_col:pd.DataFrame, diff_col:pd.DataFrame):
    """_summary_

    Args:
        filter_num_col (pd.DataFrame): corresponding filter number
        diff_col (pd.DataFrame): difference in activation between positive and negative class
    """
    # Plotting the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(filter_num_col, diff_col, marker='.', linestyle='-')
    plt.xlabel('Filter number')
    plt.ylabel('Difference in activation between positive and negative class')
    plt.xticks(filter_num_col, rotation=45)  # Set x-axis ticks to integer values with rotation
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


def draw_edge_activated_graph(edges_df:pd.DataFrame, pos_dict:pd.DataFrame):
    """Draw an individual patient graph using NetworkX highlighting which edges might 
    be most associated to the prediction.

    Args:
        edges_df (pd.DataFrame): Dataframe with information about edges including: start and end nodes, whether it's activated,
        weight, and edge_label.
        pos_dict (dictionary): dictionary with x and y coordinates for each node:(x,y)
    """
    # Convert df to list of tuples for Networkx
    edges = []
    for _, row in edges_df.iterrows():
        edge = (row['start_node'], row['end_node'], {'activated': row['activated'], 'weight': row['weight'], 'edge_label': row['time_between']})
        edges.append(edge)

    G = nx.DiGraph()
    G.add_edges_from(edges)


    # Prepare edge colors, widths, and labels based on attributes
    edge_colors = ['red' if G[u][v]['activated'] == 1 else 'grey' for u, v in G.edges()]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    edge_labels = {(u, v): G[u][v]['edge_label'] for u, v in G.edges()}

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos_dict, with_labels=True, node_size=3000, node_color="lightblue", edge_color=edge_colors, width=edge_widths, font_size=10, font_weight="bold", arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=edge_labels, font_color='blue', font_size=10, font_weight='bold')

    #plt.title("Graph Visualisation of Patients Pathway and Connections Associated to Hip Replacement")
    plt.show()



def edge_activated_graph(input_tensors:np.array, patient_number:int, filters:np.array, labels:list, verbose:bool=False, show_plot:bool=False):
    """Draw an individual patients graph with red edges where the AI model filters have given high weights, 
    representing which edges the patterns associate to the outcome the most. 

    Args:
        input_tensors (np.array): 4D array of patient graph representations (3D).
        patient_number (int): Patient to represent as a graph.
        filters (np.array): 4D array of filters from the TG-CNN model (3D).
        labels (list): Binary outcome (0 or 1) for patient e.g. no replacement or hip replacement.
        verbose (bool): Print extra statements.
        show_plot (bool): Print the maximum activation per filter plot.
    """

    # 1. Select the patient graph to draw
    patient_graph = utils.select_patient(input_tensors, 0)

    # 2. Get the maximum activation for each filter.
    mean_activation_df = max_act_diff.max_act_diff_calc('Hip Replacement', input_tensors, filters, labels, verbose=verbose, show_plot=show_plot)

    # 3. Find the filter with the maximum activation.
        # This should have the strongest effect.
    max_act_filt = utils.get_max_act_filt(mean_activation_df, filters)

    # 4. Element-wise multiplication of the max activation filter by the chosen graph
    act_graph = calculations.get_act_graph_array(patient_graph, max_act_filt)

    # 5. Get the edge features incl. start and end node, activation, weight, time between (edge value)
    edges_df = utils.create_edges_df(patient_graph, act_graph)

    # 6. Create a dataframe with the x coordinates for the nodes
    pos_df = utils.create_position_df(edges_df)

    # 7. Generate a list of possible y locations depending on the number of nodes per visit
    pos_list = utils.generate_pos_sequence(pos_df['max_codes_per_visit'].max())

    # 8. Add to the dataframe to y get coordinates for the nodes based on the number of nodes per timestep
    pos_df = utils.map_y_coord_to_node(pos_df, pos_list)

    # 9. Turn the x and y coordinates into a dictionary that can be read by NetworkX
    pos_dict = utils.create_pos_dict(pos_df)

    # 10. Draw the patient graph with the activated edges
    draw_edge_activated_graph(edges_df, pos_dict)