def threshold_func(X,Y):
    sorted = X[np.argsort(X[:, 1])]  # pass as arg the indices of the sorted 2nd column
    Y=Y[sorted]

