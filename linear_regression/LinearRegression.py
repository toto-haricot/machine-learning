"""[bla_bla_bla]
"""

import numpy as np

class LinearRegression:

    def __init__(self):
        #weights and biais
        self.w = None
        self.b = None
        #parameters used to normalize
        self.normalize_std_x = None
        self.normalize_mean_x = None
        self.normalize_std_y = None
        self.normalize_mean_y = None
        #model accuracy
        self.accuracy = None

    def fit(self, X, y, method='gradient_descent', lr=.05, n_epochs=500, display=False):

        n, d = X.shape
        errors = []

        self.w = np.random.normal(size=(d,1))
        self.b = 0

        # normalize X and y
        X, self.normalize_std_x, self.normalize_mean_x = self.normalize(X)
        y, self.normalize_std_y, self.normalize_mean_y = self.normalize(y)

        if method == 'gradient_descent':

            for i_ep in range(n_epochs):
                y_pred = self.forward(X)
                error = (1/(2*n))*np.sum((y - y_pred)**2)
                dw = (1/n)*(X.T @(y - y_pred))
                db = (1/n)*np.sum((y - y_pred))
                self.w += lr*dw
                self.b += lr*db
                errors.append(error)
                if display: print(f'Epoch {i_ep+1} error = {error}')

        elif method == 'direct':

            X_ = np.hstack([np.ones((n,1)), X])
            self.b = np.dot(self.pseudo_inverse(X_), y)[0]
            self.w = np.dot(self.pseudo_inverse(X_), y)[1:]

    def evaluate(self, x_test, y_test):
        
        x_test, _, _ = self.normalize(x_test, std=self.normalize_std_x, mean=self.normalize_mean_x)
        y_test, _, _ = self.normalize(y_test, std=self.normalize_std_y, mean=self.normalize_mean_y)
        
        n = x_test.shape[0]

        y_pred = self.forward(x_test)
        loss = (1/n)*np.sum((y_test - y_pred)**2)
        self.accuracy = loss
        return(loss)

    def forward(self, x):

        return(np.dot(x, self.w) + self.b)



    @staticmethod
    def normalize(dataset:np.array, std=None, mean=None):
        """This function normalizes a input data set stored in a numpy array. The data set is center-reduced. 
        The standart deviations and means to use for centering and reducing can be passed as arguments. If it is not the case,
        the function will compute it based on the columns values of the data set. 

        Args:
            dataset (np.array): dataset to normalize
            std (_type_, optional): standart deviation(s). Defaults to None.
            mean (_type_, optional): mean(s). Defaults to None.

        Returns:
            tuple: dataset normalized, means and std used to normalized
        """

        data = np.copy(dataset)
        n, p = data.shape
        
        if (std is not None) and (mean is not None):
            assert std.shape[0] == mean.shape[0], 'arguments std and mean should have same dimensions'
            assert dataset.shape[1] == std.shape[0], f'dataset : {dataset.shape} // std : {std.shape}'
            # assert dataset.shape[1] == std.shape[0], 'std and mean should have as many columns as dataset'
            data = np.apply_along_axis(lambda x: (x-mean)/std, 1, data)
            data = data.reshape((n,p))
            return(data, std, mean)
        
        else: 

            std_ = np.zeros((p), dtype='float')
            mean_ = np.zeros((p), dtype='float')

            for i in range(p):

                mean, std = data[:,i].mean(), data[:,i].std()
                std_[i] = std
                mean_[i] = mean
                data[:,i] = (data[:,i] - mean)/ std

            return (data, std_, mean_)
    
    @staticmethod
    def pseudo_inverse(X:np.array):
        return(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T))


    def accuracy(self):
        return self.error

