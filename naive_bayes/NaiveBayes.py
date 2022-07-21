import numpy as np

class NaiveBayes():

    def __init__(self):
        # means and standart deviations
        self.mus = None 
        self.sigmas = None
        self.classes = None
        # probabilities of each class
        self.probas = None
        # normalization parameters

    
    def fit(self, X_train:np.array, y_train:np.array):
        """This method learns the parameters that best fit to the training data, the mean
        and standart deviation for each feature of each class and also the probability of
        every class

        Args:
            X_train (np.array): input training data
            y_train (np.array): class training data
        """

        n, d = X_train.shape
        classes = np.unique(y_train)
        data_train = np.hstack([X_train, y_train])
        K = len(classes)

        self.mus = np.zeros((K,d))
        self.sigmas = np.zeros((K,d))
        self.classes = classes
        self.probas = np.zeros(K)

        for idx, c in enumerate(classes):

            data_class = data_train[data_train[:,-1] == c]
            X_class, y_class = data_class[:,:-1], data_class[:,-1]
            n_c = len(X_class)

            self.mus[idx,:] = X_class.mean(axis=0)
            self.sigmas[idx,:] = X_class.std(axis=0)
            self.probas[idx] = n_c/N

    def predict(self, X:np.array):
        


    @staticmethod
    def gaussian_pdf(x:float, sigma:float=1., mu:float=0.):
        term1 = 1/(sigma*(2*np.pi)**.5)
        term2 = ((x-mu)**2)/(2*sigma**2)
        return(term1*np.exp(-term2))






