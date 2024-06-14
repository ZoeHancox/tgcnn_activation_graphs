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

def get_max_act_filt(mean_activation_df:pd.DataFrame):
    """Using the dataframe of acitvation difference select the filter with the largest difference.

    Args:
        mean_activation_df (pd.DataFrame): dataframe with difference in activation between positive and negative class 
        for each filter.

    Returns:
        np.array: 3D array representing one filter
    """

    max_idx = mean_activation_df['Difference'].idxmax()
    max_act_filt_num = mean_activation_df.loc[max_idx, 'Filter']
    max_act_filt =  filters[max_act_filt_num-1] # minus 1 as we don't have a filter called 0
    return max_act_filt