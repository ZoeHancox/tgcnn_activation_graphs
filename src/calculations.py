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
