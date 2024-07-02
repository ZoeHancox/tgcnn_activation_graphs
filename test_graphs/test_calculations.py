import pytest # noqa: F401
import numpy as np
import pandas as pd
import os
print(os.getcwd())

from tgcnn_act_graph import utils, max_act_diff, figures, calculations  # noqa: E402

test_pats = np.array([[[[0, 0, 0], 
                          [0, 0, 3], 
                          [0, 0, 0]], [[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 4, 4]], [[0, 0, 0], 
                                                    [0, 2, 0], 
                                                    [0, 2, 0]], [[0, 0, 0], 
                                                                 [5, 0, 5], 
                                                                 [0, 0, 0]]],                        
                        [[[1, 0, 0], 
                          [1, 0, 0], 
                          [1, 0, 0]], [[0, 0, 0],
                                       [0, 8, 0],
                                       [0, 0, 0]], [[0, 11, 0], 
                                                    [0, 0, 0], 
                                                    [0, 11, 0]], [[0, 0, 30], 
                                                                 [0, 0, 30], 
                                                                 [0, 0, 30]]],                       
                          
                          [[[0, 0, 9], 
                          [0, 0, 0], 
                          [0, 0, 0]], [[0, 0, 0],
                                       [7, 0, 0],
                                       [7, 0, 0]], [[0, 0, 0], 
                                                    [0, 0, 2], 
                                                    [0, 0, 0]], [[0, 0, 0], 
                                                                 [1, 0, 0], 
                                                                 [0, 0, 0]]],                          
                          [[[1, 1, 1], 
                          [0, 0, 0], 
                          [0, 0, 0]], [[0, 0, 1],
                                       [0, 0, 1],
                                       [0, 0, 0]], [[0, 0, 0], 
                                                    [0, 1, 0], 
                                                    [0, 1, 0]], [[0, 2, 0], 
                                                                 [0, 0, 0], 
                                                                 [0, 2, 0]]]]
                        
                        )

test_filts = np.array([[[[0, 0, 1], 
                    [0, 1, 0], 
                    [0, 0, 1]], [[1, 1, 0], 
                                 [0, 0, 0], 
                                 [0, 0, 0]]], 
                   [[[0, 1, 0], 
                    [0, 1, 0], 
                    [0, 0, 1]], [[1, 0, 0], 
                                 [0, 0, 0], 
                                 [0, 0, 0]]]])


test_pat = np.array([[[0, 0, 0], 
                        [0, 0, 3], 
                        [0, 0, 0]], [[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 4, 4]], [[0, 0, 0], 
                                                [0, 2, 0], 
                                                [0, 2, 0]], [[0, 0, 0], 
                                                                [5, 0, 5], 
                                                                [0, 0, 0]]])

test_filt = np.array([[[0, 0, 1], 
                        [0, 1, 0], 
                        [0, 0, 1]], [[1, 1, 0], 
                                        [0, 0, 0], 
                                        [0, 0, 0]]])

labels = [0, 1, 1, 0] #hip_replacement labels
def test_y_coord_list_gen():
    """Test the lists generated for the y coordinates of the nodes
    are generating as expected.
    """
    auto_seq = utils.generate_pos_sequence(10)

    test_ans1 = auto_seq[0]
    true_ans1 = [0]

    assert test_ans1 == true_ans1, "The two lists should be equal"

    test_ans2 = auto_seq[4]
    true_ans2 = [-2, -1, 0, 1, 2]

    assert test_ans2 == true_ans2, "The two lists should be equal"

    test_ans3 = auto_seq[7]
    true_ans3 = [-4, -3, -2, -1, 1, 2, 3, 4]

    assert test_ans3 == true_ans3, "The two lists should be equal"


def test_pos_dict():
    """ Test that the y coordinate dictionaries have the same keys and values.
    The dictionaries may not match exactly as the y coordinate is dependent on the order of the
    DataFrame.
    """
    # How pos dictionary should look:
    pos_true = {
        '1_v0': (0, 0),
        '2_v1': (1, 0),
        '1_v2': (2, 1),
        '2_v2': (2, -1),
        '1_v3': (3, 0),
        '0_v4': (4, 1),
        '2_v4': (4, -1)
    }

    # Code chunk to get y position dictionary
    mean_activation_df = max_act_diff.max_act_diff_calc('Hip Replacement', test_pats, test_filts, labels, verbose=False, show_plot=False)
    max_act_filt = utils.get_max_act_filt(mean_activation_df, test_filts)
    act_graph = calculations.get_act_graph_array(test_pat, max_act_filt)
    edges_df = utils.create_edges_df(test_pat, act_graph)
    pos_df = utils.create_position_df(edges_df)
    pos_list = utils.generate_pos_sequence(pos_df['max_codes_per_visit'].max())
    pos_df = utils.map_y_coord_to_node(pos_df, pos_list)
    pos_test = utils.create_pos_dict(pos_df)


    pos_true_keys = set(pos_true.keys())
    pos_test_keys = set(pos_test.keys())

    pos_true_values = set(pos_true.values())
    pos_test_values = set(pos_test.values())

    assert pos_true_keys == pos_test_keys, "The keys of dictionaries of pos_true and pos_test should be equal"
    assert pos_true_values == pos_test_values, "The values of dictionaries of pos_true and pos_test should be equal"


def test_edge_weight():
    """Test that the weight of the activated edge in the test patient is calculated correctly.
    """

    true_weight1 = 2.0
    
    mean_activation_df = max_act_diff.max_act_diff_calc('Hip Replacement', test_pats, test_filts, labels, verbose=False, show_plot=False)
    max_act_filt = utils.get_max_act_filt(mean_activation_df, test_filts)
    act_graph = calculations.get_act_graph_array(test_pat, max_act_filt)
    edges_df = utils.create_edges_df(test_pat, act_graph)

    test_weight1 = edges_df.loc[edges_df['activated']==1, 'weight'].astype(float).values

    assert true_weight1 == test_weight1, "The weight of this edge should be 2."

def test_extract_visit_num():
    """Test the extracted visit number is the visit number and not the node number.
    """
    string = "5_v2"
    true_visit_num = 2
    false_visit_num = 5
    test_visit_num = utils.extract_visit_number(string)

    assert true_visit_num == test_visit_num, "True visit number should equal test visit number"
    assert false_visit_num != test_visit_num, "True visit number should equal test visit number not EHR code number"


def test_repeat_array_fractional():
    """Test to make sure filter is repeating correctly.
    """
    repeat = 2.5
    test_repeat_array_fractional = calculations.repeat_array_fractional(test_filt, repeat)
    true_repeated_filter = np.array([[[0, 0, 1], 
                                    [0, 1, 0], 
                                    [0, 0, 1]], [[1, 1, 0], 
                                                [0, 0, 0], 
                                                [0, 0, 0]], [[0, 0, 1], 
                                                            [0, 1, 0], 
                                                            [0, 0, 1]], [[1, 1, 0], 
                                                                        [0, 0, 0], 
                                                                        [0, 0, 0]], [[0, 0, 1], 
                                                                                    [0, 1, 0], 
                                                                                    [0, 0, 1]]])
    assert np.array_equal(true_repeated_filter, test_repeat_array_fractional), "The filter should be 5 timesteps"    


