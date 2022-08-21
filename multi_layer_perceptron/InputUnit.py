

class InputUnit():

    def __init__(self, data):
        self.z = 0
        self.data = data

    def forward(self,i):
        self.z = self.data[i]
        return(self.data[i])

    def plug(self,v):
        v.preceding.append(self)
        v.npr += 1