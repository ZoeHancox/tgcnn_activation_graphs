import numpy as np
import pandas as pd

def extract_visit_number(s):
    """Extract the visit number from a string

    Args:
        s (str): string to split

    Returns:
        int: visit number
    """
    return int(s.split('_')[1][1:])

def get_max_act_filt(mean_activation_df:pd.DataFrame, filters:np.array):
    """Using the dataframe of acitvation difference select the filter with the largest difference.

    Args:
        mean_activation_df (pd.DataFrame): dataframe with difference in activation between positive and negative class 
        for each filter.
        filters (np.array): 4D array with 3D filters.

    Returns:
        np.array: 3D array representing one filter.
    """

    max_idx = mean_activation_df['Difference'].idxmax()
    max_act_filt_num = mean_activation_df.loc[max_idx, 'Filter']
    max_act_filt =  filters[max_act_filt_num-1] # minus 1 as we don't have a filter called 0
    return max_act_filt



def create_edges_df(patient_graph, act_graph):
    """Create a DataFrame of the edges of the patient graph, including the start and end nodes
    whether the edge is activated (more than 0), the 'weight' of the activation, and the time 
    between visits.

    Args:
        patient_graph (np.array): 3D numpy array showing the patients health codes over time.
        act_graph (np.array): 3D numpy array showing graph activation.

    Returns:
        DataFrame: with columns start_node, end_node, activated, weight, time_between.
    """
    num_edges = np.count_nonzero(patient_graph)
    edges_df = pd.DataFrame(columns=['start_node', 'end_node', 'activated', 'weight', 'time_between'], index=range(num_edges))
    num_nodes = patient_graph.shape[1]
    timesteps = patient_graph.shape[0]

    row_num = 0
    for t in range(timesteps):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if patient_graph[t, i, j] != 0:
                    if t == 0:
                        start_node_v = 0
                        end_node_v = 1
                    else:
                        start_node_v = t
                        end_node_v = t+1
                    
                    edges_df.at[row_num, 'end_node'] = f'{j}_v{end_node_v}' #[row num, col name]
                    edges_df.at[row_num, 'weight'] = act_graph[t, i, j]
                    edges_df.at[row_num, 'time_between'] = patient_graph[t, i, j] #add a column for the edge label too (time between from original graph)
                    edges_df.at[row_num, 'start_node'] = f'{i}_v{start_node_v}' #[row num, col name]
                    
                    row_num += 1


    # activated_graph col is more than 0, the activated column = 1, otherwise it is 0
    edges_df['activated'] = edges_df['weight'].apply(lambda x: 0 if x == 0 else 1)  
    # change 0 weights to 0.5 so we can still see them on the graph figure
    edges_df['weight'] = edges_df['weight'].apply(lambda x: 0.5 if x == 0 else x)
    return edges_df