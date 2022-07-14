"""This module is meant to provide useful functions for dataset manipulations during machine learning projects
"""

import numpy as np

def split_dataset(data_X:np.array, data_y:np.array, prop:float=.8):
    """This function will split X and y into training and testing sets after randomly shuffle. 

    Args:
        data_X (np.array): input feature
        data_y (np.array): expected output
        prop (float, optional): Proportion of samples to put into training set. Defaults to .8.
    """
    
    assert data_X.shape[0] == data_y.shape[0], 'X and y must have same lengths'

    if len(data_y.shape) == 1: 

        n, d = data_y.shape[0], 1
        data_y = data_y.reshape((n,d))

    data = np.hstack([data_X, data_y])
    
    np.random.shuffle(data)
    
    n_train = int(prop*n)
    
    X_train, y_train = data[:n_train, :-d], data[:n_train, -d:]
    X_test, y_test = data[n_train:, :-d], data[n_train:, -d:]

    return(X_train, y_train, X_test, y_test)
