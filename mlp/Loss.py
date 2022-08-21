

class Loss():

    def __init__(self, y):
        self.npr = 0
        self.preceding = []
        self.z = 0
        self.y = y

    def forward(self, i):
        inputs = np.zeros((self.npr))
        for j in range(self.npr):
            inputs[j] = self.preceding[j].forward(i)
        output = (np.sum((inputs - self.y[i])**2))**.5
        self.z = output
        return(output)