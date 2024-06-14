import numpy as np


# TODO TEST HERE FOR CHECKING WHETHER A FRACTIONAL DIVISION PRODUCES A CORRECT ARRAY

def repeat_array_fractional(array: np.ndarray, repeats: float):
    """
    Repeats the filter array a specified non-integer number of times.
    This is because the graph may divide fractionally compared to the filter size.
    
    Parameters:
    array (np.ndarray): The array to repeat.
    repeats (float): The number of times to repeat the array.
    
    Returns:
    np.ndarray: The repeated array.
    """
    # Separate the integer and fractional parts
    int_part = int(repeats)
    frac_part = repeats - int_part

    # Repeat the array for the integer part
    repeated_array_int = np.tile(array, (int_part, 1, 1))

    # Calculate the number of rows to take for the fractional part
    num_rows = array.shape[0]
    frac_rows = int(num_rows * frac_part)

    # Take the fractional part from the array
    if frac_rows > 0:
        repeated_array_frac = array[:frac_rows, :, :]
        # Concatenate the integer part and the fractional part
        result_array = np.concatenate((repeated_array_int, repeated_array_frac), axis=0)
    else:
        result_array = repeated_array_int

    return result_array

def leaky_relu(x:float, alpha:int):
    """If the activation is negative multiply it by alpha,
    otherwise retain the alpha value.
    This is to avoid 0 gradients.

    Args:
        x (float): activation sum
        alpha (int): hyperparameter to select or tune

    Returns:
        float: activation sum with leaky ReLU applied
    """
    return max(alpha * x, x) # if it's negative multiply by 0.01


def get_act_graph_array(pat_graph: np.array, max_act_filt: np.array):
    """Get the activated_graph which is the graph element-wise multiplied by the 'sliding window'. 

    The stride must be the same as the filter.shape[0] so that slices don't overlap (otherwise we
    get multiple mappings).

    The filter is repeated to be the same length as the patient graph then the two graphs are 
    multiplied together (element-wise) to get the activated graph.

    Raises:
        ValueError: if the patient graph is not 3D then raise an error.

    Returns:
        np.array: filter array .* patient array.
    """

    if pat_graph.ndim != 3:
        raise ValueError("The input array must be 3-dimensional.")

    vectorized_leaky_relu = np.vectorize(leaky_relu)
    
    filt_num_repeats = int(pat_graph.shape[0]/max_act_filt.shape[0])


    # To reduce compute repeat the filter to be the same size as the patient graph.
    # That way the two can be directly multiplied rather than sliding.
    filt_repeated = repeat_array_fractional(max_act_filt, filt_num_repeats)

    # Element-wise multiplication for the patient graph and the repeated filter
    filt_times_graph = filt_repeated * pat_graph

    # apply a leaky relu (activation)
    activated_graph = vectorized_leaky_relu(filt_times_graph, 0.01)

    return activated_graph