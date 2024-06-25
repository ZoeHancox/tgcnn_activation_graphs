# Create Activation Graphs from the TG-CNN Model

## About the Project

![Tests](https://github.com/ZoeHancox/tgcnn_activation_graphs/actions/workflows/tests.yml/badge.svg)

Produce graphs using the 3D CNN layers from the trained [TG-CNN model](https://dl.acm.org/doi/10.1007/978-3-031-16564-1_34). These graphs show which edges or timesteps are the most important during model prediction.

> [!IMPORTANT]
> _Please note that the examples provided in this repository are fictitious and do not contain any real patient data._  



### Activation mapping graphs for edges steps:

The GIF below shows how the filters from the 3D CNN layer are used to show edge activation per input graph:

![Edge activation graph](documentation/edge_activation_graph.gif)

1. Extract the filters from the 3D CNN layer of the TG-CNN model.
2. Find the filter with the strongest differentiation of maximum activation between the positive and negative class amongst all the input graphs.
3. Select the filter with the largest activation difference to show edge activation.
4. Make the stride length the same as the number of timesteps in the filter.
5. Do element-wise multiplication between the filter and the input graph, to get the edge activation tensor.
6. Use the edge activation tensor to get weights for the edges. Edges with zero activation are grey, edges with activation are red. The higher the edge activation weight the thicker the edge.
7. Observe which edges affect the prediction outcome. Nodes are named as e.g. 5_t1 (Read Code = 5, timestep = 1).

![Edge activation graph plot from code](documentation/edge_activation_graph_output.png)




---
### Equation

To get the edge weights: For a given filter $f_{k}$, patient $p$, and time step $i$, the process can be summarised as:

1. Extract the slice:

   $W_{i}^{(p, k)}$ = $G_{p}[i:i+F, :, :]$,
   
   
   where $F$ is the size of the filter in the time dimension, $G_{p}$ as the input tensor for patient $p$, $f_{k}$ as the $k$-th filter, and $W_{i}^{(p, k)}$ is the slice of the tensor $G_{p}$ at time step $i$ with the same dimensions as the filter $f_{k}$.

2. Apply the filter and sum the result:
   
   $S_{i}^{(p, k)}$ = $\sum (W_{i}^{(p, k)} \odot f_{k}),$

   where $\odot$ denotes element-wise multiplication.

3. Apply the leaky ReLU activation function:
   
   $A_{i}^{(p, k)}$ = $leaky ReLU(S_{i}^{(p, k)}, \alpha),$

    where $\alpha$ is the leaky ReLU parameter.

4. Compute the maximum activation value ($M$) across all time steps:
   
   $M_{(p, k)}$ = $\max_{i} (A_{i}^{(p, k)})$

5. Combine these into one formula:

    $M_{(p, k)}$ = $\max_{i} \left( leaky ReLU \left( \sum (G_{p}[i:i+F, :, :] \odot f_{k}), \alpha \right) \right)$

## PROJECT STRUCTURE

The main code is found in the `tgcnn_act_graph` folder of the repository. See Usage below for more information.

```
├── documentation             # Images and other background files
├── test_graphs               # Tests for tgcnn_act_graph
├── tgcnn_act_graph           # Source files
├── create_graphs.ipynb       # Example of how to use this code locally
├── LICENSE.txt
├── README.md
└── requirements.txt          # Which packages are required to run this code
```

### BUILT WITH
[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
- [NumPy](https://numpy.org/)
- [NetworkX](https://networkx.org/)

### Getting Started and Installation


#### To clone the repo:

To clone this repository:
- Open Git Bash, your Command Prompt or Powershell
- Navigate to the directory where you want to clone this repository: `cd/path/to/directory`
- Run git clone command:
`git clone https://github.com/ZoeHancox/tgcnn_activation_graphs`

To create a suitable environment we suggest using Anaconda:
- Build conda environment: `conda create --name graph_viz python=3.8`
- Activate environment: `conda activate graph_viz`
- Install requirements: `python -m pip install -r ./requirements.txt`

---

#### To install the package:

- Build conda environment: `conda create --name graph_viz python=3.8`
- Activate environment: `conda activate graph_viz`
- Install package: `pip install tgcnn-act-graph`


## USAGE

See examples in `create_graphs.ipynb` for how to use this code.

If you have installed the package you can:

```
from tgcnn_act_graph.figures import edge_activated_graph
import numpy as np

# Load or create your 4D filters
filters = np.array([[[[0, 0], 
                      [0, 1]], [[1, 1], 
                                [0, 0]]], 
                     [[[0, 1], 
                        [0, 1]], [[1, 0], 
                                 [0, 0]]]])

# Load or create your 4D patient graphs
input_tensors = np.array([[[[0, 0], 
                          [0, 3]], [[0, 0],
                                    [4, 0]], [[8, 8], 
                                             [0, 0]]],                        
                        [[[7, 0], 
                          [0, 0]], [[9, 0],
                                    [0, 0]], [[4, 4], 
                                             [0, 0]]]])


labels = [0, 1] # positive or negative labels

edge_activated_graph(input_tensors=input_tensors, patient_number=1,  filters=filters, labels=labels, verbose=False, show_plot=False)
```

## ROADMAP

Features to come:

- [x] Show edge activation using NetworkX
- [ ] Add a list of your own node names rather than using ints
- [ ] Show time step activation using NetworkX

## SUPPORT

See the [Issues](https://github.com/ZoeHancox/tgcnn_activation_graphs/issues) in GitHub for a list of proposed features and known issues. Contact [Zoe Hancox](mailto:Z.L.Hancox@Leeds.ac.uk) for further support. 


## TESTING

Run tests by using `pytest test_graphs/test_calculations.py` in the top directory.

## LICENSE

Unless stated otherwise, the codebase is released under the BSD Licence. This covers both the codebase and any sample code in the documentation.

See [LICENCE](https://github.com/ZoeHancox/tgcnn_activation_graphs/blob/main/LICENSE.txt) for more information.

## ACKNOWLEDGEMENTS

The TG-CNN model was developed using data provided by patients and collected by the NHS as part of their care and support. 