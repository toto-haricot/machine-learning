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
        predictions = self.give_class(X_train, centroids)

        keep_going = True

        while keep_going:

            dataset = np.hstack([X_train, predictions])
            centroids = self.search_centroids(dataset)
            new_predictions = self.give_class(X_train, centroids)

            if changes(predictions, new_predictions) == 0: keep_going=False

            predictions = new_predictions

        return(predictions)




    def search_centroids(self, X_:np.array):

        n, d = X_.shape
        classes = np.uniques(X_[:,-1])
        new_centroids = np.zeros((self.K, d-1))

        for i,c in enumerate(classes):
            
            subset = X_[X_[:,-1] == c]
            centroid = np.mean(subset, axis=0)
            new_centroids[i] = centroid

        return new_centroids




    @staticmethod
    def give_class(X_train:np.array, K_centroids:np.array):
        """For each data point of X_train we compute the closest centroid based on
        Euclidean distance metrics

        Args:
            X_train (np.array): data points
            K_centroids (np.array): K centroid points
        """

        n, d = X_train.shape
        K = len(K_centroids)

        X_train_3d = np.reshape(X_train, (n, 1, d))
        X_train_3d = np.repeat(X_train_3d, repeats=K, axis=1)

        norm = np.abs(X_train_3d - K_centroids)**2
        norm = np.sum(norm, axis=2)

        pred = np.argmin(norm, axis=1)
        pred = np.reshape(pred, (n,1))

        return(pred)

    
    @staticmethod
    def changes(new_classes:np.array, old_classes:np.array):

        return(np.sum(new_classes == old_classes))









