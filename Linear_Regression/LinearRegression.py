"""[module description]
"""

from cv2 import normalize
import numpy as np
#import matplotlib.pyplot as plt

class LinearRegression:
    """[summary]
    """
    def __init__(self):
        #list with the weights
        self.w = []
        #the biais
        self.b = 0
        self.error = 0

    def normalize(self, dataset):
        """To normalize a numpy dataset column wise"""
        data = np.copy(dataset)
        for i in range(data.shape[1]):
            col_mean = data[:,i].mean()
            col_std = data[:,i].std()
            data[:,i] = (data[:,i] - col_mean)/col_std
        return data

    def fit(self, X, y, lr=.005, n_epochs=500):
        """Training method to set the parameters to optimal value closer 

        Args:
            X ([array]): the training data set
            y ([array]): targets
            lr (float, optional): Learning rate. Defaults to .005.
            n_epochs (int, optional): [description]. Defaults to 500.
        """
        n, d = X.shape
        X = normalize(X)

        errors = []

        self.w = np.random.normal(size=(d,1))
        self.b = 0

        #training loop
        for _ in range(n_epochs):
            y_pred = np.dot(X,self.w) + self.b
            error = (1/(2*n))*np.sum(np.sqrt(y - y_pred))
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
