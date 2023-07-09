#!/usr/bin/env python3
#
# Definition of a function in a separate file (Gram-Schmidt orthogonalization)
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023

# module imports
import numpy as np

def GramSchmidt(X):
    """
    Gram-Schmidt process for orthonormalizing a set of vectors.

    Usage: iret,U = GramSchmidt(X)

    Inputs:   X is a 2D ndarray object
    Outputs:  iret is a success (0) or failure (1) flag
              U is a 2D ndarray of the same shape as X, upon
                  successful completion, the columns of U form an
                  orthonormal basis for the column-space of X.
    """

    # copy input matrix
    U = X.copy()

    # check that there is work to do
    if (np.size(U,1) < 1):
        return [0, U]

    # get entry magnitude (for linear dependence check)
    Xmax = np.linalg.norm(X,np.inf)

    # normalize first column
    colnorm = np.linalg.norm(U[:,0])
    if (colnorm < 1.e-13*Xmax):
        print("GramSchmidt error: vectors are linearly-dependent!")
        return [1, U]
    U[:,0] /= colnorm

    # iterate over remaining vectors, performing Gram-Schmidt process
    for i in range(1,np.size(U,1)):

        # subtract off portions in directions of existing basis vectors
        for j in range(i):
            U[:,i] -= (np.dot(U[:,i], U[:,j]) * U[:,j])

        # normalize vector, checking for linear dependence
        colnorm = np.linalg.norm(U[:,i])
        if (colnorm < 1.e-13*Xmax):
            print("GramSchmidt error: vectors are linearly-dependent!")
            return [1, U]
        U[:,i] /= colnorm

    return [0,U]
