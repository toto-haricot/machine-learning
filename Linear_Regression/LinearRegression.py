import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self):
        #list with the weights
        self.w = []
        #the biais
        self.b = 0
        
    def fit(self, X, y, lr=.005, n_epochs=500):
        X = np.array(X)
        X = self.normalize() #function to define, it normalizes the data
        
