import numpy as np

### Unfolding (matricization)
def unfold(X, mode_number):
    """
    Unfolding (matricization)

    Parameters
    ----------
    X : ndarray
        Input tensor.
    mode_number : int
        Mode number along which to unfold.
        ``numpy's axis = (mode_number - 1)``

    Returns
    -------
    Y : ndarray
        Unfolded (Matrixed) tensor.
    """
    axis = mode_number - 1     # numpy's axis = (mode_number - 1)
    N = np.ndim(X)             # order of input tensor
    L = np.size(X)             # total number of input tensor elements
    col_len = np.size(X, axis) # column Length of output matrix
    row_len = int(L / col_len) # row Length of output matrix
    ### mode (axis) transpose
    permute = [axis] + list(range(0, axis)) + list(range(axis+1, N))
    X_transposed = np.transpose(X, permute)
    ### unfolding 
    unfolding_shape = (col_len, row_len)
    Y = np.reshape(X_transposed, unfolding_shape)
    return Y

### Folding (tensorization)
def fold(Y, mode_number, shape):
    """
    Unfolding (matricization)

    Parameters
    ----------
    Y : ndarray
        Input matrix.
    mode_number : int
        Mode number along which to fold.
        ``numpy's axis = (mode_number - 1)``
    shape : tuple or list of ints
        shape of folding (tensorization).

    Returns
    -------
    X : ndarray
        Folded (tensorized) matrix.
    """
    axis = mode_number - 1 # numpy's axis = (mode_number - 1)
    N = len(shape)         # order of output tensor
    ### folding
    permute = [axis] + list(range(0, axis)) + list(range(axis+1, N))
    folding_shape = []
    for i in permute: folding_shape.append(shape[i])
    Y_folded = np.reshape(Y, folding_shape)
    ### mode (axis) transpose
    if mode_number==1: permute_inv = list(range(0, N))
    else: permute_inv = list(range(1, axis+1)) + [0] + list(range(axis+1, N))
    X = np.transpose(Y_folded, permute_inv)
    return X