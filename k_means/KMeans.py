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
            display (Bool): True or False if you want to plot the clusters for each
                            iteration of K-Means
        """
        
        self.K = K
        n_epoch = 1
        keep_going = True
        n, d = X_train.shape
        X_train = self.normalize(X_train)

        if d > 3: display = False

        # random select K points as first centroids
        random_idx = random.sample(range(n), K)
        initial_centroids = X_train[random_idx]
        self.centroids = initial_centroids

        predictions = self.give_class(X_train, initial_centroids)

        while keep_going:

            dataset = np.hstack([X_train, predictions])
            # compute new centroids coordinates
            centroids = self.search_centroids(dataset)
            self.centroids = centroids
            # find new clusters
            new_predictions = self.give_class(X_train, centroids)
            new_dataset = np.hstack([X_train, new_predictions])
            # compute number of data points changed class
            changes = self.changes(predictions, new_predictions)
            # plotting clustering evolution
            if display: self.plot_clusters(dataset, centroids, dim=d, epoch=n_epoch)
            # checking if optimum found
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

    
    @staticmethod
    def plot_clusters(dataset:np.array, centroids:np.array, dim:int, fig_size=(12,7), epoch=None):

        n, d = dataset.shape

        if d == 3:

            plt.figure(figsize=fig_size)
            plt.title(f"Epoch {epoch-1}")
            plt.scatter(x=dataset[:,0], y=dataset[:,1], c=dataset[:,2])
            plt.scatter(x=centroids[:,0], y=centroids[:,1], c='r')
            plt.show()

        if d == 4: 

            plt.figure(figsize=fig_size)
            ax = plt.axes(projection='3d')
            ax.set_title(f"Epoch {epoch-1}")
            ax.scatter(xs=dataset[:,0], ys=dataset[:,1], zs=dataset[:,2], c=dataset[:,3])
            plt.show()

    @staticmethod
    def normalize(dataset:np.array):

        n, d = dataset.shape
        dataset_norm = np.zeros((n, d))

        for i in range(d):

            mean, std = dataset[:,i].mean(), dataset[:,i].std()
            dataset_norm[:,i] = (dataset[:,i] - mean) / std

        return(dataset_norm)















