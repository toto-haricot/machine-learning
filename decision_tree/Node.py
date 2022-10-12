
class Node():
    """This class represent a decision tree node.

    Parameters:
    -----------

    """

    def __init__(self, attribute=None, threshold=None, value=None, left_branch=None, 
                right_branch=None):
        # decision parameters
        self.decision_attribute = attribute
        self.decision_threshold = threshold
        self.value = value
        # child nodes
        self.child_left = left_branch
        self.child_right = right_branch


    def find_decision(self):

        split = {"information_gain" : -1, "attribute" : None, "threshold" : None}

        sub_dataset = DecisionTree.dataset[self.indices, :]

        for feature in sub_dataset.shape[1]:

            thresholds = np.unique(sub_dataset[:, feature])

            for thresh in thresholds:

                left = sub_dataset[sub_dataset[:,feature] <= thresh][:,-1]
                right = sub_dataset[sub_dataset[:,feature] > thresh][:,-1]

                n_l, n_r = len(left), len(right)
                n = n_l + n_r

                entropy_left = n_l/n*self.entropy(left)
                entropy_right = n_r/n*self.entropy(right)

                information_gain = self.impurity - entropy_left - entropy_right

                if information_gain > split["information_gain"]:

                    split["information_gain"] = information_gain
                    split["attribute"] = feature
                    split["threshold"] = thresh

        return(split["attribute"], split["threshold"])


    def is_leaf(self):
        
        global min_samples
        global max_depth

        impurity = self.impurity()

        if (len(self.indices) <= min_samples or
            self.depth >= max_depth or
            impurity == 0):

            node_class = np.argmax(DecisionTree.dataset[indices, -1])
            self.leaf = node_class
            return(node_class)


    def impurity(self):

        self.impurity = self.entropy(DecisionTree.dataset[self.indices, -1])

        return(self.impurity)


    @staticmethod
    def entropy(X:np.array):
        freq = np.bincount(X)
        freq = freq / len(freq)
        entropy = -np.sum([p * np.log2(p) for p in freq if p > 0])
        return(entropy)


    @staticmethod
    def split(X:np.array, col:int, threshold):
        left_idx = [i for i,x in enumerate(X[:,col] <= threshold)]
        right_idx = [i for i,x in enumerate(X[:,col] > threshold)]
        return(left_idx, right_idx)



    def grow(self):

        if self.is_leaf() is: return None

        self.decision_attribute, self.decision_threshold = self.find_decision()

        left_node = Node()
        right_node = Node()

        self.child_left = left_node
        self.child_right = right_node

        setattr(left_node, 'parent', self)
        setattr(right_node, 'parent', self)

        left_idx, right_idx = self.split(DecisionTree.dataset[self.indices], self.decision_attribute, self.decision_threshold)

        setattr(left_node, 'indices', left_idx)
        setattr(right_node, 'indices', right_idx)

        setattr(left_node, 'depth', self.depth+1)
        setattr(right_node, 'depth', self.depth+1)

        return(left_node, left_node)
        



        


