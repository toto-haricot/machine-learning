import numpy as np

import Node

from utils import divide_by_feature

class DecisionTree():

    def __init__(self, depth_max=1e3, min_impurity=1e-7, min_sample_size=2):

        self.depth_max = depth_max
        self.min_impurity = min_impurity
        self.min_sample = min_sample_size
        self.root = None 
        self.impurity = None

    def fit()

    def build_tree(self, X, y, current_depth=0):

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        n, d = np.shape(X)
        Xy = np.concatenate((X,y), axis=1)

        if (n > self.min_sample) and (current_depth <= self.max_depth):

            for j in range(d):

                feature_values = np.expand_dims(X[:, j], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:

                    Xy1, Xy2 = divide_by_feature(Xy, )



        

