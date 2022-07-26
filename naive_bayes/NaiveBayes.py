import numpy as np

class NaiveBayes():

    def __init__(self):
        # data dimension
        self.d = None
        self.K = None
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

        self.d = d
        self.K = K
        self.mus = np.zeros((K,d))
        self.sigmas = np.zeros((K,d))
        self.classes = classes
        self.probas = np.zeros(K)

        for idx, c in enumerate(classes):

            data_class = data_train[data_train[:,-1] == c]
            X_class, y_class = data_class[:,:-1], data_class[:,-1]
            n_c = len(X_class)

            self.mus[idx,:] = np.mean(X_class.astype(float), axis=0)
            self.sigmas[idx,:] = np.std(X_class.astype(float),axis=0)
            self.probas[idx] = n_c/n


    def classify(self, X:np.array):
        """This method gives the class predictions for a single sample

        Args:
            X (np.array): numpy array of dimension one 

        Returns:
            class_label: label of the class predictedd by the model for single input X
        """
        proba_class = np.zeros((self.d)) 
        for k in range(self.K):
            proba_k = self.probas[k]
            for (i,x) in enumerate(X):
                proba_k *= self.gaussian_pdf(x, sigma=self.sigmas[k,i], mu=self.mus[k,i])
            proba_class[k] = proba_k
        return self.classes[np.argmax(proba_class)]


    def predict(self, X:np.array):
        """This method gives the vector of predictions for an input array X with several observations

        Args:
            X (np.array): input array
        """
        predictions = []
        for (i,X_) in enumerate(X):
            predictions.append(self.classify(X_))
        predictions = np.array(predictions).reshape((len(X), 1))
        predictions = predictions.reshape((len(X), 1))
        return(predictions)

    
    def evaluate(self, X_test:np.array, y_test:np.array):
        """Computes the model accuracy. Should be applied on X_test and y_test

        Args:
            X_test (np.array): testing inputs
            y_test (np.array): testing taget classes
        """
        accuracy = np.mean(self.predict(X_test) == y_test)
        self.accuracy = accuracy
        return(f"Model accuracy : {round(accuracy, 4)}")


    @staticmethod
    def gaussian_pdf(x:float, sigma:float=1., mu:float=0.):
        eps = 1e-4
        term1 = 1/(sigma*(2*np.pi)**.5 + eps)
        term2 = ((x-mu)**2)/(2*sigma**2)
        return(term1*np.exp(-term2))






