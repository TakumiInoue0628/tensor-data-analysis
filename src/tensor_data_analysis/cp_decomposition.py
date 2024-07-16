import numpy as np
import tqdm

### Move to parent directory
from os.path import dirname, abspath
import sys
parent_dir = dirname(abspath(__file__))
sys.path.append(parent_dir)

from calculation import *
from reconstruction import *

### CP decomposition using ALS(alternating least squares)
def cp_decomposition_als(Tensor, cp_rank, max_iter=10, initialize_type='random', solution_type='lstsq', non_negative=False, random_seed=0):
    """
    CP decomposition using ALS(alternating least squares)

    Parameters
    ----------
    Tensor : ndarray
        Input tensor.\n
    cp_rank : int
        CP rank (tensor rank).\n
    max_iter : int
        Maximum number of update iterations.\n
    initialize_type : str
        Type of initialize factor vectors.\n
    solution_type : str
        Type of solving linear-matrix-equation.\n
        - 'normal' : using ``'numpy.linalg.solve'``\n
        - 'lstsq' : using ``'numpy.linalg.lstsq'`` (least squares solution)\n
    non_negative : bool
        Non-negative constraint.\n
    random_seed : int
        Seed of random number generation (default is ``0``).\n

    Returns
    -------
    Matrices_list : list ``[ndarray, ndarray, ndarray,...]``
        Factor matrices.\n
    Tensor_reconst : ndarray
        Reconstructed Tensor.\n
    cost_history : ndarray
        Cost transition.\n
    """
    np.random.seed(seed=random_seed)
    
    N = np.ndim(Tensor)             # order of input tensor
    tensor_shape = np.shape(Tensor) # tensor's shape

    ### initialize factor matrices
    Matrices_list = []
    if initialize_type=='zeros': 
        for size in tensor_shape: Matrices_list.append(np.zeros((size, cp_rank)))
    if initialize_type=='ones': 
        for size in tensor_shape: Matrices_list.append(np.ones((size, cp_rank)))
    if initialize_type=='random': 
        for size in tensor_shape: Matrices_list.append(np.random.random((size, cp_rank)))

    cost = []
    for _ in tqdm.tqdm(range(max_iter), desc="CP-decomposition (ALS)", leave=False):

        for i in range(N):
            mode_number = i + 1

            ### caluculate H{i}
            H_i = np.ones((cp_rank, cp_rank))
            for j in range(N):
                if j==i: continue # i=j is skip
                H_i *= np.matmul(Matrices_list[j].T, Matrices_list[j])

            ### caluculate G{i}
            G_i = np.zeros((tensor_shape[i], cp_rank))
            for j in range(cp_rank):
                ### get vector list
                Vector_list = []
                for Matrices in Matrices_list: Vector_list.append(Matrices[:, j])
                ### caluculate G{i}
                G_i[:, j] = np.reshape(leave_one_out_prod_tensor_vector(Tensor, Vector_list, mode_number), tensor_shape[i])

            ### caluculate A{i} H{i} = G{i}

            ### normal
            if solution_type=='normal': A_i = np.linalg.solve(H_i.T, G_i.T)
            ### using least squares solution
            if solution_type=='lstsq': A_i, _, _, _ = np.linalg.lstsq(H_i.T, G_i.T, rcond=None)

            A_i = A_i.T

            ### non-negative constraint
            if non_negative:
                A_i = np.maximum(A_i, 0)

            Matrices_list[i] = A_i

        ### tensor reconstruction
        Tensor_reconst = reconst_tensor_cp(Matrices_list)
        ### cost
        cost.append((np.linalg.norm(Tensor) - np.linalg.norm(Tensor_reconst))**2)
    cost_history = np.array(cost)

    return Matrices_list, Tensor_reconst, cost_history