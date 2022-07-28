import random
import numpy as np

class KMeans():

    def __init__(self):

        # model parameters
        self.K = None
        self.centroids = None

        # evaluation on labeled data
        self.accuracy = None

    def fit(self, X_train:np.array, K:int):
        
        self.K = K
        n, d = X_train.shape

        # random select K points
        random_idx = random.sample(range(n), K)
        centroids = X_train[random_idx]


    def give_class(self, X_train:np.array, K_centroids:np.array):
        """For each data point of X_train we compute the closest centroid based on
        Euclidean distance metrics

        Args:
            X_train (np.array): data points
            K_centroids (np.array): K centroid points
        """

        n, d = X_train.shape
        K = len(K_centroids)

        X_train_3d = np.reshape(X_train, (n, 1, d))
        X_train_3d = np.repeat(X_train_3d, repeats=K)

        norm = np.abs(X_train - K_centroids)**2
        norm = np.sum(norm, axis=2)

        pred = np.argmin(norm, axis=1)

        return(pred)








