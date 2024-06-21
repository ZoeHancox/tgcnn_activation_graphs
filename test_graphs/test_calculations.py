import pytest

from src import utils, max_act_diff, figures, calculations


# test_pats =
# test_filts = 


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


# How pos dictionary should look:

# pos = {
#     '1_v0': (0, 0),
#     '2_v1': (1, 0),
#     '1_v2': (2, 1),
#     '2_v2': (2, -1),
#     '1_v3': (3, 0),
#     '0_v4': (4, 1),
#     '2_v4': (4, -1)
# }

# How edges list should look:

# edges = [
#     ('1_v0', '2_v1', {'activated': 0, 'weight': 0.5, 'edge_label': 3}),
#     ('2_v1', '1_v2', {'activated': 0, 'weight': 0.5, 'edge_label': 4}),
#     ('2_v1', '2_v2', {'activated': 0, 'weight': 0.5, 'edge_label': 4}),
#     ('1_v2', '1_v3', {'activated': 1, 'weight': 2, 'edge_label': 2}),
#     ('2_v2', '1_v3', {'activated': 0, 'weight': 0.5, 'edge_label': 2}),
#     ('1_v3', '0_v4', {'activated': 0, 'weight': 0.5, 'edge_label': 5}),
#     ('1_v3', '2_v4', {'activated': 0, 'weight': 0.5, 'edge_label': 5})
# ]