"""class for building linear regression models
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
        """method to train the model at giving better predictions. Concretely is will adjust weights
        and biais with gradient descent to minimize the loss

        Args:
            X (np.array): input training set
            y (np.array): output training set
            method (str, optional): training method. Can be 'gradient_descent' (default) or 'direct'.
            lr (float, optional): learning rate. Defaults to .05.
            n_epochs (int, optional): number of epochs in training loop. Defaults to 500.
            display (bool, optional): select True to have updates on the error for each epoch. Defaults to False.
        """

        n, d = X.shape
        errors = []

        self.w = np.random.normal(size=(d,1))
        self.b = np.random.normal()

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

    def evaluate(self, x_test:np.array, y_test:np.array):
        """method to evaluate the model accuracy on the testing, which is normalized with same parameters as 
        training set (mean and standart deviation)

        Args:
            x_test (np.array): input testing set
            y_test (np.array): output testing set
        """
        x_test, _, _ = self.normalize(x_test, std=self.normalize_std_x, mean=self.normalize_mean_x)
        y_test, _, _ = self.normalize(y_test, std=self.normalize_std_y, mean=self.normalize_mean_y)
        
        n = x_test.shape[0]

        y_pred = self.forward(x_test)
        loss = (1/n)*np.sum((y_test - y_pred)**2)
        self.accuracy = loss
        return(loss)

    def forward(self, x):
        """simple method to get the prediction of the model for a set of inputs

        Args:
            x (np.array): input set
        """
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
        """function to compute the pseudo inverse of a matrix. We can note that if X is inversible, the pseudo
        inverse is equal to the inverse matrix

        Args:
            X (np.array): any matrix to inverse
        """
        return(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T))


    def accuracy(self):
        return self.error

