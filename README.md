# Create Activation Graphs for TG-CNN Model

Produce graphs from the 3D CNN layers from the trained TG-CNN model. These graphs should which edges or timesteps are the most important during model prediction. 


The equations of how maximum activation is calculated is as follows:

To get the edge weights: For a given filter $f_{k}$, patient $p$, and time step $i$, the process can be summarised as:

1. Extract the slice:

   $
   W_{i}^{(p, k)} = G_{p}[i:i+F, :, :],
   $
   
   where $F$ is the size of the filter in the time dimension, $G_{p}$ as the input tensor for patient $p$, $f_{k}$ as the $k$-th filter, and $W_{i}^{(p, k)}$ is the slice of the tensor $G_{p}$ at time step $i$ with the same dimensions as the filter $f_{k}$.

2. Apply the filter and sum the result:
   
   $
   S_{i}^{(p, k)} = \sum (W_{i}^{(p, k)} \odot f_{k}),
   $

   where $\odot$ denotes element-wise multiplication.

3. Apply the leaky ReLU activation function:
   
   $
   A_{i}^{(p, k)} = leaky ReLU(S_{i}^{(p, k)}, \alpha),
   $

    where $\alpha$ is the leaky ReLU parameter.

4. Compute the maximum activation value across all time steps:
   
   $
   \text{max\_activation\_value}_{p, k} = \max_{i} (A_{i}^{(p, k)})
   $

5. Combine these into one formula:

    $
    Max Activation Value_{p, k} = \max_{i} \left( leaky ReLU \left( \sum (G_{p}[i:i+F, :, :] \odot f_{k}), \alpha \right) \right)
    $
