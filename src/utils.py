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


def create_position_df(edges_df):
    """Create a DataFrame to find the number of nodes per visit.

    Args:
        edges_df (pd.DataFrame): DataFrame with columns start_node, end_node, activated, weight, time_between.

    Returns:
        pd.DataFrame: with columns showing node name (node), visit number (x), node number in visits (cumulative count),
        and maximum codes per visit.
    """
    pos_df = edges_df[['start_node', 'end_node']].stack().drop_duplicates().reset_index(drop=True)
    pos_df = pos_df.to_frame(name='node')
    pos_df['x'] = pos_df['node'].apply(extract_visit_number)
    pos_df['cumulative_count'] = pos_df.groupby('x').cumcount()
    pos_df['max_codes_per_visit'] = pos_df.groupby('x')['cumulative_count'].transform('max') + 1

    return pos_df


def generate_pos_sequence(x):
    """Generate a list of lists to get y coordinate positions for the nodes
     based on the number of events recorded per visit.

    Args:
        x (int): maximum number of nodes in any one visit

    Returns:
        List: List of lists of y coordinates index mapping to the max nodes for
               each visit.
    """
    sequence = []
    for i in range(x):
        if i % 2 == 0:  # Even index, include zero
            sequence.append(list(range(-i // 2, i // 2 + 1)))
        else:  # Odd index, exclude zero
            sublist = list(range(-(i // 2 + 1), i // 2 + 2))
            sublist.remove(0)
            sequence.append(sublist)
    return sequence


def get_pos_y_value_per_node(row, pos_list):
    """Get the y position for each node. Use max_codes_per_visit column to select the sublist 
    and the cumulative_count to get the position from the sublist.

    Args:
        row (pd.Series): row of the DataFrame

    Returns:
        int: y coordinate position 
    """
    cum_count = row['cumulative_count']
    max_codes = row['max_codes_per_visit']
    return pos_list[max_codes - 1][cum_count]

def map_y_coord_to_node(pos_df, pos_list):
    """Map the y coordinates to the relevant node and correct row.

    Args:
        pos_df (pd.DataFrame): columns with node name and x coordinate.
        pos_list (list of lists): List of lists of y coordinates index mapping to the max nodes for
               each visit.

    Returns:
        pd.DataFrame: dataframe with x and y coordinates for node plotting.
    """
    pos_df['y'] = pos_df.apply(lambda row: get_pos_y_value_per_node(row, pos_list), axis=1)
    return pos_df

def create_pos_dict(pos_df):
    """Make a dictionary with the node name as the key and the x and y coordinates 
    as a tuple value.

    Args:
        pos_df (pd.DataFrame): dataframe with columns for the node name, x coordinates,
        and y coordinates.

    Returns:
        dict: dictionary of node: (x,y)
    """
    # the visit number is x and the y value is the number of nodes with the same visit number
    pos = pos_df.set_index('node')[['x', 'y']].apply(tuple, axis=1).to_dict()
    return pos