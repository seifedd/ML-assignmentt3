import numpy as np


def threshold_func(X,Y):
    sorted = X[np.argsort(X[:, 1])]  # pass as arg the indices of the sorted 2nd column
    Y=Y[sorted]
    indices_where_change = np.where(Y[:-1] != Y[1:])[0]
    return  indices_where_change



