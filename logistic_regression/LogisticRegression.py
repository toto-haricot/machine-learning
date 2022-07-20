"""class for building logistic regression models
"""

import numpy as np

class LogisticRegression():

    def __init__(self):
        #weights and biais
        self.w = None
        self.b = None
        #parameters used to normalize
        self.normalize_std_x = None
        self.normalize_mean_x = None
        #performances
        self.accuracy = 0

    def fit(self, X, y, normalize=False, n_epoch=500, lr=.05):
        """This function implements gradient descent for learning the weights and biais that minimize the loss function

        Args:
            X (np.array): inputs
            y (np.array): labels
            normalize (bool, optional): If the dataset is already normalized turn to True. Defaults to False.
            n_epoch (int, optional): Number of epochs for the gradient descent. Defaults to 500.
            lr (float, optional): Learning rate. Defaults to .05.

        Returns:
            list: records of the training loss for each epoch
        """

        if not normalize: 
            
            X, self.normalize_std_x, self.normalize_mean_x = self.normalize(X)

        n, d = X.shape

        self.w = np.random.normal(size=(d,1))
        self.b = np.random.normal()

        errors = []

        print(f'Start training for {n_epoch} epochs')

        #start the gradient descent
        for i in range(n_epoch):

            lc = (np.dot(X, self.w) + self.b)
            y_pred = self.sigmoid(lc)
            dw = (1/d)*np.dot(X.T, (y_pred - y))
            db = (1/d)*np.sum((y_pred - y))
            self.w -= lr*dw
            self.b -= lr*db
            error = self.loss(y_pred, y)
            errors.append(error)

        return errors

    def predict(self, X):
        """This function outputs the prediction given by the model for an input X. 
        Note that X should be normalized beforehand.

        Args:
            X (np.array): input 
        """
        probas = self.sigmoid(np.dot(X, self.w) + self.b)
        y_pred = (probas>.5)*1
        return(y_pred)

    def evaluate(self, X, y, normalize=True):
        """This function will evaluate the model accuracy on the dataset passed in argument. We suggest to pass in X_test 
        and y_test to evaluate accuracy on the testing dataset. 

        Args:
            X (np.array): testing input
            y (np.array): testing labels
            normalize (bool, optional): If the dataset is already normalized turn to True. Defaults to False.

        Returns:
            float: accuracy
        """

        if not normalize:

            X, _, _ = self.normalize(X, std=self.normalize_std_x, mean=self.normalize_mean_x)

        y_pred = self.predict(X)
        n_pred = y.shape[0]
        accuracy = np.sum(y_pred == y)/n_pred
        self.accuracy = accuracy
        return(accuracy)


    @staticmethod
    def sigmoid(x):
        return(1/(1+np.exp(-x)))

    @staticmethod
    def loss(y_pred, y_true):
        m = len(y_pred)
        assert len(y_pred) == len(y_true), 'predictions and target should have same lengths'
        return(-np.mean(y_true*(np.log(y_pred)) - (1-y_true)*(np.log(1-y_pred))))

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

