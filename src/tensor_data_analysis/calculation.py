import numpy as np
from transformation import *

### n-th mode product of Tensor and Matrix
def n_mode_prod_tensor_matrix(Tensor, Matrix, mode_number):
    """
    n-th mode product of Tensor and Matrix

    Parameters
    ----------
    Tensor : ndarray
        Input tensor.
    Matrix : ndarray
        Input matrix.
    mode_number : int
        Mode number to calculate the product.
        ``numpy's axis = (mode_number - 1)``

    Returns
    -------
    Y : ndarray
        Producted tensor.
    """
    axis = mode_number - 1          # numpy's axis = (mode_number - 1)
    tensor_shape = np.shape(Tensor) # tensor's shape
    ### new tensor's shape
    new_tensor_shape = np.copy(tensor_shape)
    new_tensor_shape[axis] = np.size(Matrix, 0)
    ### unfolding
    Tensor_unfolded = unfold(Tensor, mode_number)
    ### Matrix product of Tensor and Matrix
    matrix_product_TandM = np.matmul(Matrix, Tensor_unfolded)
    ### Folding
    Y = fold(matrix_product_TandM, mode_number, new_tensor_shape)
    return Y

### all-mode product of Tensor and Matrix
def all_mode_prod_tensor_matrix(Tensor, Matrix_list):
    """
    All mode product of Tensor and Matrix

    Parameters
    ----------
    Tensor : ndarray
        Input tensor.
    Matrix_list : list ``[ndarray, ndarray, ndarray,...]``
        Input matrices list.

    Returns
    -------
    Y : ndarray
        Producted tensor.
    """
    ### all-mode product of Tensor and Matrix
    Y = np.copy(Tensor)
    for axis, Matrix in enumerate(Matrix_list):
        ### n-th mode product of Tensor and Matrix
        mode_number = axis + 1
        Y = n_mode_prod_tensor_matrix(Y, Matrix, mode_number)
    return Y

### all-mode product of Tensor and Vector
def all_mode_prod_tensor_vector(Tensor, Vector_list):
    """
    All mode product of Tensor and Vector

    Parameters
    ----------
    Tensor : ndarray
        Input tensor.
    Vector_list : list ``[ndarray, ndarray, ndarray,...]``
        Input  vectors list.

    Returns
    -------
    Y : ndarray
        Producted tensor.
    """
    ### matrixing of vectors
    ### shape(n,) --> shape(n, 1)
    Matrix_list = []
    for vector in Vector_list: Matrix_list.append(np.reshape(vector, (np.size(vector, 0), 1)))
    ### all-mode product of Tensor and Vector
    Y = np.copy(Tensor)
    for axis, Matrix in enumerate(Matrix_list):
        ### n-th mode product of Tensor and Matrix
        mode_number = axis + 1
        Y = n_mode_prod_tensor_matrix(Y, Matrix.T, mode_number)
    return Y


### leave-one-out product of Tensor and Vector
def leave_one_out_prod_tensor_vector(Tensor, Vector_list, mode_number_leave_one_out):
    """
    Leave-one-out product of Tensor and Vector

    Parameters
    ----------
    Tensor : ndarray
        Input tensor.
    Vector_list : list ``[ndarray, ndarray, ndarray,...]``
        Input  vectors list.
    mode_number_leave_one_out : int
        mode number of leave one out

    Returns
    -------
    Y : ndarray
        Producted tensor.
    """
    ### matrixing of vectors
    ### shape(n,) --> shape(n, 1)
    Matrix_list = []
    for vector in Vector_list: Matrix_list.append(np.reshape(vector, (np.size(vector, 0), 1)))
    ### all-mode product of Tensor and Vector
    Y = np.copy(Tensor)
    for axis, Matrix in enumerate(Matrix_list):
        mode_number = axis + 1
        ### leave one out
        if mode_number==mode_number_leave_one_out: continue
        ### n-th mode product of Tensor and Matrix
        Y = n_mode_prod_tensor_matrix(Y, Matrix.T, mode_number)
    return Y
