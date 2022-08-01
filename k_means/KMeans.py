import random
import numpy as np

import matplotlib.pyplot as plt

class KMeans():

    def __init__(self):
        # model parameters
        self.K = None
        self.centroids = None
        # evaluation on labeled data
        self.accuracy = None


    def fit(self, X_train:np.array, K:int, n_epochs_max=200, display=True):
        """This method will look for K clusters in the dataset X_train following
        K-Means algorithm method. 

        Args:
            X_train (np.array): training dataset
            K (int): number of clusters to look for
        """
        
        self.K = K
        n, d = X_train.shape
        n_epoch = 1

        # random select K points
        random_idx = random.sample(range(n), K)
        initial_centroids = X_train[random_idx]

        self.centroids = initial_centroids

        predictions = self.give_class(X_train, initial_centroids)
        dataset = np.hstack([X_train, predictions])

        keep_going = True

        while keep_going:

            dataset = np.hstack([X_train, predictions])

            centroids = self.search_centroids(dataset)
            self.centroids = centroids
            new_predictions = self.give_class(X_train, centroids)

            new_dataset = np.hstack([X_train, new_predictions])
            changes = self.changes(predictions, new_predictions)

            if display:
                plt.figure(figsize=(12, 7))
                plt.title(f"Epoch {n_epoch-1}")
                plt.scatter(x=dataset[:,0], y=dataset[:,1], c=dataset[:,2])
                plt.scatter(x=centroids[:,0], y=centroids[:,1], c='r')
                plt.show()

            if (n_epoch >= n_epochs_max) or (changes == 0): keep_going=False

            predictions = new_predictions
            n_epoch += 1

        print(f"End of clustering\n\t{n_epoch} epochs done \n\t{changes} changes on last epochs ")
        return(dataset)


    def search_centroids(self, X_:np.array):
        """This function will compute the coordinates of the centroids for each class
        of the input dataset. 

        Args:
            X_ (np.array): dataset to cluster

        Returns:
            np.array: centroids coordinates 
        """
        n, d = X_.shape
        classes = np.unique(X_[:,-1])
        new_centroids = np.zeros((self.K, d-1))

        for i,c in enumerate(classes):
            
            subset = X_[X_[:,-1] == c,:-1]
            centroid = np.mean(subset, axis=0)
            new_centroids[i] = centroid

        return new_centroids


    @staticmethod
    def give_class(X_train:np.array, K_centroids:np.array):
        """For each data point of X_train we compute the closest centroid based on
        Euclidean distance metrics

        Args:
            X_train (np.array): data points
            K_centroids (np.array): coordinates of the centroids
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
        """This static method returns the number of changes between the two input
        vector new_classes and old_classes

        Returns: 
            int: number of different values
        """
        return(np.sum(new_classes != old_classes))









