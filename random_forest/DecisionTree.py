
import Node

class DecisionTree():

    def __init__(self):
        
        # caracteristics
        self.nodes = []
        self.depth = None
        self.accuracy = None

        # stopping criteria


    def fit(self, X_train, y_train, stop_growing=False):

        global dataset

        dataset = np.hstack([X_train, y_train])
        
        root = Node()
        root.root = True
        root.depth = 0
        root.indices = np.arange(len(dataset))
        
        self.nodes.append(root)


        while stop_growing == False:

            new_nodes = []
            only_leafs = True

            for node in self.nodes[-1]:

                child_nodes = node.grow()

                if child_nodes: 
                    only_leafs = False
                    new_nodes.append(child_nodes)

            self.nodes.append(new_nodes)
            stop_growing = only_leafs



