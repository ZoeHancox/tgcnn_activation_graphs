import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

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


def draw_edge_activated_graph(edges_df, pos_dict):
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