import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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