
import numpy as np
import pandas as pd

class LDA:

    def __init__(self):
        # data statitics 
        self.mu = None
        self.pi = None
        self.sigma = None
        self.inv_sigma = None
        # data description
        self.classes = None
        # performances
        self.accuracy = None


    def fit(self, X_train:np.array, y_train:np.array):
        """This method will compute the means, the covariances and the probabilities of each class so 
        that we then have all elements required to compute the discriminative functions

        Args:
            X_train (np.array): input training set
            y_train (np.array): output training set
        """
        
        n, d = X_train.shape
        classes = np.unique(y_train)
        data_train = np.hstack([X_train, y_train])

        self.mu = np.zeros((len(classes), d))
        self.pi = np.zeros(len(classes))
        self.sigma = np.zeros((d,d))
        self.classes = classes

        all_sigmas = np.zeros((len(classes), d, d))

        for idx, c in enumerate(classes):

            data_class = data_train[data_train[:,-1] == c]
            X_class, y_class = data_class[:,:-1], data_class[:,-1]

            n_c = len(data_class)
            pi_c = n_c / n
            mu_c = np.zeros(d)

            for j in range(d):

                mu_c[j] = X_class[:,j].mean()

            sigma_c = np.dot(np.transpose(X_class - mu_c), X_class - mu_c)/n_c
            all_sigmas[idx,:,:] = sigma_c

            self.mu[idx, :] = mu_c
            self.pi[idx] = pi_c
            
        self.sigma = np.sum(all_sigmas, axis=0)
        self.inv_sigma = np.linalg.inv(np.sum(all_sigmas, axis=0))


    def predict(self, X:np.array):
        """This method predicts the classes of all given input X

        Args:
            X (np.array): inputs to predict class

        Returns:
            np.array: class predictions for input X
        """

        term1 = np.dot(np.dot(X, self.inv_sigma), np.transpose(self.mu))
        term2 = .5*np.diagonal(np.dot(np.dot(self.mu, self.inv_sigma), np.transpose(self.mu)))
        term3 = np.log(self.pi)

        pred = np.argmax(term1 - term2 + term3, axis=1)
        pred = pred.reshape((len(X),1))
        pred = self.classes[pred]

        return(pred)

    def evaluate(self, X_test:np.array, y_test:np.array):
        """This method computes the accuracy of the model on testing set

        Args:
            X_test (np.array): inputs testing set
            y_test (np.array): outputs testing set
        """
        pred = self.predict(X_test)
        accuracy = np.sum(pred == y_test) / len(X_test)
        self.accuracy = accuracy
        return(f"Accuracy is equal to : {round(accuracy, 4)}")









