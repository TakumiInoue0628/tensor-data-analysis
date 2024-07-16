import numpy as np
from calculation import *
from reconstruction import *

### rank-1 decomposition
def rank_one_decomposition_als(Tensor, max_iter=30, initialize_type='ones'):
    """
    Rank-1 decomposition using ALS(altenating least squares)

    Parameters
    ----------
    Tensor : ndarray
        Input tensor.
    max_iter : int
        Maximum number of update iterations.
    initialize_type : str
        Type of initialize factor vectors.

    Returns
    -------
    scale_cofficient : int
        Scale cofficient of the factor vectors.
    Vector_list : list ``[ndarray, ndarray, ndarray,...]``
        Factor vectors.
    Tensor_reconst : ndarray
        Reconstructed Tensor.
    cost_history : ndarray
        Cost transition.
    """
    
    N = np.ndim(Tensor)             # order of input tensor
    tensor_shape = np.shape(Tensor) # tensor's shape

    ### initialize
    Vector_list = []
    if initialize_type=='ones': 
        for size in tensor_shape: Vector_list.append(np.ones((size)))
    if initialize_type=='random': 
        for size in tensor_shape: Vector_list.append(np.random.random((size)))

    cost = []
    for _ in range(max_iter):

        for axis, _ in enumerate(Vector_list):
            ### leave-one-out product of Tensor and Vector
            mode_number = axis + 1
            Y = leave_one_out_prod_tensor_vector(Tensor, Vector_list, mode_number)
            ### vectorization
            y = np.reshape(Y, np.size(Y))
            ### normalize
            Vector_list[axis] = y / np.linalg.norm(y)

        ### scale coefficient
        lam = all_mode_prod_tensor_vector(Tensor, Vector_list)
        ### tensor reconstruction
        Tensor_reconst = lam * reconst_rank_one_tensor(Vector_list)
        ### cost
        cost.append((np.linalg.norm(Tensor) - np.linalg.norm(Tensor_reconst))**2)

    scale_cofficient = np.reshape(lam, 1)
    cost_history = np.array(cost)

    return scale_cofficient, Vector_list, Tensor_reconst, cost_history