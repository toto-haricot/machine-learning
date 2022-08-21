import numpy as np

class NeuralUnit():

    def __init__(self, k, u):
        #coordinates
        self.k = k
        self.u = u
        #weigths and biais
        self.w = []
        self.b = 0
        #units around
        self.preceding = []
        self.following = []
        self.npr = 0
        #backpropagation
        self.delta = []
        self.w = []
        self.b = 0
        #output
        self.z = 0


    def set_parameters(self):
        self.w = np.random.normal(size=self.npr)
        self.b = np.random.normal()
        

    def forward(self, i):
        inputs = np.zeros((self.npr))
        for j in range(self.npr):
            inputs[j] = self.preceding[j].forward(i)
        output = self.sigmoid(np.dot(inputs, self.w)+self.b)
        self.z = output
        return(output)


    def plug(self, v):
        self.following.append(v)
        self.nfo += 1
        v.preceding.append(self)
        n.npr += 1


    def backprop(self, i, deltas):

        self.delta = np.zeros(self.w.shape)
        self.w_grad = np.zeros(self.w.shape)
        self.b_grad = 0 
        
        for j in range(npr):
            self.delta[j] = self.z * (1 - self.z) * self.w[j] * deltas[self.u-1]
            self.w_grad[j] = self.z * (1 - self.z) * self.preceding[j].z * deltas[self.u-1]
            
        self.b_grad = self.z * (1 - self.z) * deltas[self.u-1]

    
    @staticmethod
    def sigmoid(x):
        res = 1 / (1 + np.exp(-x)) 
        return res

        
