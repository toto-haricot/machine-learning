"""[module description]
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """model for Linear Regression 
    """
    def __init__(self):
        #list with the weights
        self.w = []
        #the biais
        self.b = 0
        self.error = 0
        #parameters used to normalize
        self.normalize_std = []
        self.normalize_mean = []

    def fit(self, X, y, lr=.005, n_epochs=500):
        """Training method to set the weights closer to optimal values

        Args:
            X ([array]): the training data set
            y ([array]): the training targets
            lr (float, optional): Learning rate. Defaults to .005.
            n_epochs (int, optional): [description]. Defaults to 500.
        """
        n, d = X.shape

        errors = []

        self.w = np.random.normal(size=(d,1))
        self.b = 0

        #training loop
        for _ in range(n_epochs):
            y_pred = np.dot(X,self.w) + self.b
            error = (1/(2*n))*np.sum((y - y_pred)**2)
            dw = (1/n)*np.dot(X.T ,y - y_pred)
            db = (1/n)*np.sum((y - y_pred))
            self.w -= lr*dw
            self.b -= lr*db
            self.error = error
            errors.append(error)
        return errors

    def predict(self, X):
        """outputs the prediction
        """
        X = normalize(X)
        y_pred = np.dot(X.T, self.w)
        return y_pred

    def accuracy(self):
        return self.error

    def normalize(self, dataset:np.array):
        """_summary_

        Args:
            dataset (np.array): _description_

        Returns:
            np.array: dataset normalized
        """
        data = np.copy(dataset)
        for i in range(data.shape[1]):
            mean, std = data[:,i].mean(), data[:,i].std()
            self.normalize_std.append(std)
            self.normalize_mean.append(mean)
            data[:,i] = (data[:,i] - mean)/ std
        return (data[:,:-1], data[:,-1])
