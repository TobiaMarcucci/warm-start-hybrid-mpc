# external imports
import numpy as np

def sym2mat(x, expr):
    '''
    Extracts from the symbolic affine expression the matrices such that expr(x) = A x + b.

    Parameters
    ----------
    x : sympy matrix of sympy symbols
        Variables of the affine expression.
    expr : sympy matrix of sympy symbolic affine expressions
        Left hand side of the inequality constraint.

    Returns
    -------
    jacobian : np.array
        Jacobian of the affine expression.
    offset : np.array
        Offset term of the affine expression.
    '''

    # left hand side
    jacobian = np.array(expr.jacobian(x)).astype(np.float64)

    # offset term
    offset = np.array(expr.subs({xi:0 for xi in x})).astype(np.float64).flatten()
    
    return jacobian, offset

def unpack_bmat(A, indices, direction):
    '''
    Unpacks a matrix in blocks.

    Parameters
    ----------
    A : np.array
        Matrix to be unpacked.
    indices : list of int
        Set of indices at which the matrix has to be cut.
    direction : string
        'h' to unpack horizontally and 'v' to unpack vertically.

    Returns
    -------
    blocks : list of np.array
        Blocks extracted from the matrix A.
    '''

    # initialize blocks
    blocks = []

    # unpack
    i = 0
    for j in indices:

        # horizontally
        if direction == 'h':
            blocks.append(A[:,i:i+j])

        # vertically
        elif direction == 'v':
            blocks.append(A[i:i+j,:])

        # raise error if uknown key
        else:
            raise ValueError('Unknown unpacking direction %s.'%direction)

        # increase index by j
        i += j

    return blocks