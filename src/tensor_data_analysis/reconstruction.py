import numpy as np
from calculation import *

### reconstruct of rank-1 tensor
def reconst_rank_one_tensor(factor_vector_list):
    """
    Reconstruct of rank-1 tensor (outer product of factor vectors)

    Parameters
    ----------
    factor_vector_list : list ``[ndarray, ndarray, ndarray,...]``
        Input factor vectors list.

    Returns
    -------
    Tensor : ndarray
        Rank-1 tensor.
    """
    N = len(factor_vector_list) # total number of the factor vectors
    ### matrixing of vectors
    ### shape(n,) --> shape(n, 1)
    factor_matrix_list = []
    for vector in factor_vector_list: factor_matrix_list.append(np.reshape(vector, (np.size(vector, 0), 1)))
    ### all-mode product of Tensor and Matrix
    ### (outer product of factor vectors)
    Tensor = all_mode_prod_tensor_matrix(np.ones([1]*N), factor_matrix_list)
    return Tensor

### Reconstruct of tensor base CP-decomposition
def reconst_tensor_cp(factor_matrices_list):
    """
    econstruct of tensor base CP-decomposition

    Parameters
    ----------
    factor_matrices_list : list \n
        Input factor vectors list.\n
        ``[A, B, C,...]`` ( ``A`` is ndarray of shape(I, R) )\n
        I is vector's size, and R is rank.

    Returns
    -------
    Tensor : ndarray
        Tensor.
    """
    R = np.size(factor_matrices_list[0], 1) # rank of each factor matrices
    ### tensor's shape
    tensor_shape = []
    for factor_matrix in factor_matrices_list: tensor_shape.append(np.size(factor_matrix, 0))
    ### sum of rank1-tensors
    Tensor = np.zeros(tensor_shape)
    for r in range(R):
        rank = r + 1
        ### get rank-1 factor vectors from each factor matrices
        factor_vector_list = []
        for factor_matrix in factor_matrices_list: factor_vector_list.append(factor_matrix[:, r])
        ### reconstruct of rank-1 tensor (outer product of factor vectors)
        Tensor += reconst_rank_one_tensor(factor_vector_list)
    return Tensor