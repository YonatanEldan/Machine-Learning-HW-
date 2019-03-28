import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # if the data is empty then gini is 0
    if(len(data)==0): return 0
    length = (len(data[0,:]))
    arr = data[:,length-1]
    S = len(arr)
    #creates a dictionary where labels are keys and number of labels in data are values 
    unique, counts = np.unique(arr, return_counts=True)
    d = dict(zip(unique, counts))
    for key, value in d.items(): 
        gini += ((value/S)**2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (1 - gini)

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # if the data is empty then entropy is 0
    if(len(data)==0): return 0
    length = (len(data[0,:]))
    arr = data[:,length-1]
    S = len(arr)
    #creates a dictionary where labels are keys and number of labels in data are values 
    unique, counts = np.unique(arr, return_counts=True)
    d = dict(zip(unique, counts))
    for key, value in d.items(): 
        entropy += ((value/S)*(np.log2(value/S)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (entropy*(-1))

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        
    def add_child(self, node):
        self.children.append(node)
        
    def best_threshold(self , featureIndex , data, impurity):
        labelColumnIndex = (len(data[0,:]))
        featuredata = data[:, [featureIndex, labelColumnIndex-1]]
        # might be useless
        otherNode = DecisionNode(featureIndex,0)
        firstChild = DecisionNode(featureIndex,0)
        secondChild = DecisionNode(featureIndex,0)
        otherNode.add_child(firstChild)
        otherNode.add_child(secondChild)
        # till here
        sortedArr = np.sort(data[:,featureIndex])
        possibleThresholds = []
        for i in range(len(sortedArr)-1):
            possibleThresholds.append((sortedArr[i]+sortedArr[i+1])/2)
           
        bestThreshold = possibleThresholds[0]
        bestThresholdGain = -1
        for currentThreshold in possibleThresholds:
            print("the best bestThreshold is:", bestThreshold)
            print("the best bestThreshold gain is:", bestThresholdGain)
            currentGain = self.calc_information_gain(otherNode, impurity, featureIndex, currentThreshold, featuredata) 
            if (currentGain > bestThresholdGain):
                bestThreshold = currentThreshold
                bestThresholdGain = currentGain 
        return(bestThreshold)
        

    def best_feature(self,data,impurity):
        length = (len(data[0,:]))
        arr = data[:,length-1]
        S = len(arr)    
            
    
    def calc_information_gain(self, root,impurity, featureIndex, threshold, data):
        parentGain = impurity(data)
        S = len(data)
        b = data[data[:,0]<threshold]
        c = data[data[:,0]>=threshold]
        childrensGain = ((len(b)/S)*(impurity(b))+(len(c)/S)*(impurity(c)))
        return (parentGain - childrensGain)
        
            
        
        


def build_tree(data, impurity):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

    

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def print_tree(node):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
