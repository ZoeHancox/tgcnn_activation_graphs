import pandas as pd
import numpy as np
from src import calculations, figures

def create_max_act_df(class_name:str, pat_graphs:np.array, filters:np.array, labels:list, verbose:bool):
    """Calculate the maximum activation from each filter on each patient graph. Running a filter over
    each patient graph and getting the max. 
    Assumes a stride length of one.

    Args:
        class_name (str): name to describe prediction outcome.
        pat_graphs (np.array): 4D array containing x 3D patient graphs.
        filters (np.array): 4D array containing x filters.
        labels (list): list of binary values representing positive or negative outcomes.
        verbose (bool): print or not to print extra dataframes or print statements.

    Raises:
        ValueError: if numpy array isn't 4D.

    Returns:
        pd.DataFrame: dataframe with columns for filter number, maximum activation and chosen class_name string.
    """
    if pat_graphs.ndim != 4:
        raise ValueError("The input array must be 4-dimensional.") # 4D to have multiple patients 3D graph representations
    if filters.ndim != 4:
        raise ValueError("The filters array must be 4-dimensional.") # 4D to have multiple 3D filters

    columns = [class_name, 'Max Activation', 'Filter']
    time_steps = pat_graphs.shape[1]
    num_pats = pat_graphs.shape[0]
    num_filters = filters.shape[0]
    
    # Construct empty DataFrame
    num_rows = num_pats * num_filters
    df_index = 0
    df = pd.DataFrame(columns=columns, index=range(num_rows))

    for filter_num in range(0, num_filters):
        f = filters[filter_num]
        
        # first run every patient for one filter
        for patient_num in range(0, num_pats):
            df.iloc[df_index, 0] = labels[patient_num]
            input_tensor = pat_graphs[patient_num]
            # Run a single filter over the graph
            max_activation_value = 0
            # this might need to change when the stride isn't 1 - TODO add this as an action to github 
            for i in range(0, time_steps-1): 
                first_slice = input_tensor[i:f.shape[0]+i, :, :]
                filter_times_slice = first_slice * f
                window_sum = np.sum(filter_times_slice)
                activation = calculations.leaky_relu(window_sum, 0.01) 
                if activation > max_activation_value:
                    max_activation_value = activation
            df.iloc[df_index, 1] = max_activation_value
            df.iloc[df_index, 2] = filter_num+1
            df_index += 1

            if verbose:
                print(f"The maximum activation for patient {patient_num} and filter {filter_num} = {max_activation_value}")
    
    return df

def max_act_diff_calc(class_name:str, pat_graphs:np.array, filters:np.array, labels:list, verbose:bool):
    """Calculate the max difference between the classes for each filter and return in a dataframe.

    Args:
        class_name (str): name to describe prediction outcome.
        pat_graphs (np.array): 4D array containing x 3D patient graphs.
        filters (np.array): 4D array containing x filters.
        labels (list): list of binary values representing positive or negative outcomes.
        verbose (bool): print or not to print extra dataframes or print statements.

    Returns:
        pd.DataFrame: difference between graph activation in both classes.
    """

    max_act_per_filt_df = create_max_act_df(class_name, pat_graphs, filters, labels, verbose)
    mean_activation = max_act_per_filt_df.groupby(['Filter', class_name])['Max Activation'].mean()
    mean_activation_df = mean_activation.to_frame()
    mean_activation_df.reset_index(inplace=True)

    # Calculate the difference between the two class type rows for each Filter
    mean_activation_df['Difference'] = mean_activation_df.groupby('Filter')['Max Activation'].diff()
    mean_activation_df['Difference'] = mean_activation_df['Difference'].abs()

    mean_activation_df = mean_activation_df[['Filter', 'Difference']].dropna().reset_index(drop=True)

    mean_activation_df['Filter'] = mean_activation_df['Filter'].astype(int)
    mean_activation_df['Difference'] = mean_activation_df['Difference'].astype(float)

    if verbose:
        print(mean_activation_df)

    figures.plot_activation_difference(mean_activation_df['Filter'], mean_activation_df['Difference'])

    return mean_activation_df

