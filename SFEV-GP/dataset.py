import pandas as pd
import numpy as np

def getData(path, skiprows):
    arr = np.loadtxt(path,dtype=np.float,skiprows=skiprows)
    y = []

    for item in arr:
        y.append(item[-1])

    y = np.array(y)
    n_samples = len(y)
    n_features = arr.shape[1] - 1
    X = arr[:, :n_features].reshape(n_samples,n_features)

    return X, y, n_features