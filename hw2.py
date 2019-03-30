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
    
    def __init__(self, feature=0, value=0):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.prediction = None # added a prediction field which will be updated only for nodes that are leaves at the tree
        self.children = []
        self.isLeaf = False
        
    def add_child(self, node):
        self.children.append(node)

    
    #the function calculates the optimal threshold for a feature
       
    def best_threshold(self , featureIndex , data, impurity):
        # sort the array in order to create an array with optional Thresholds
        sortedArr = np.sort(data[:,0])
        possibleThresholds = []
        
        for i in range(len(sortedArr)-1):
            possibleThresholds.append((sortedArr[i]+sortedArr[i+1])/2)
           
        bestThreshold = possibleThresholds[0]
        bestThresholdGain = -1
        for currentThreshold in possibleThresholds:
            currentGain = self.calc_information_gain(impurity, featureIndex, currentThreshold, data) 
            # the best threshhold will be the one that has the highest informaion gain
            if (currentGain > bestThresholdGain):
                bestThreshold = currentThreshold
                bestThresholdGain = currentGain 
        return (bestThreshold,bestThresholdGain)
        
            
    #the function calculates the information gain for a specific split 
    def calc_information_gain(self, impurity, featureIndex, threshold, data):
        parentGain = impurity(data)
        S = len(data)
        b = data[data[:,0]>threshold]
        c = data[data[:,0]<=threshold]
        childrensGain = ((len(b)/S)*(impurity(b))+(len(c)/S)*(impurity(c)))
        return (parentGain - childrensGain)

    # the function calculates the best feature and threshold for the data  
    def best_feature(self,data,impurity):
        bestGain = -1
        # we check for each feature who has the highest information gain - (-1) is since we don't check the labels.
        for i in range(len(data[0,:])-1):
            #create data specific for the feature
            featuredata = data[:, [i, len(data[0,:])-1]]   
            threshold, informationGain =  self.best_threshold(i,featuredata,impurity)  
            if(informationGain>bestGain):
                bestGain = informationGain 
                bestPair = (i,threshold)
            
        
        self.feature, self.value = bestPair        


    def split_by_threshold(self, data):
        a = data[data[:,self.feature]>self.value]
        b = data[data[:,self.feature]<=self.value]
        return (a,b)


def make_a_split(data, data0 , data1,chi_value):
    if chi_value==1 : return True
    Pf = np.array([(data0[:, -1] == 0).sum(), (data1[:, -1] == 0).sum()])
    Nf = np.array([(data0[:, -1] == 1).sum(), (data1[:, -1] == 1).sum()])
    Df = Pf + Nf
    

    unique, counts = np.unique(data, return_counts=True)
    d = dict(zip(unique, counts))
    P0 = d[0]/(len(data))
    P1 = d[1]/(len(data))
    E0 = P0*Df
    E1 = P1*Df
        
    return ((((np.square(Pf - E0) / E0) +  (np.square(Nf - E1) / E1)).sum())>chi_table[chi_value])

def build_tree(data, impurity, chi_value = 1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = DecisionNode()
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    NodeQueue = [root]
    DataQueue = [data]
    while ((len(NodeQueue))>0):
        curNode = NodeQueue.pop(0)
        curNodeData = DataQueue.pop(0)
        curNode.prediction = calc_prediction(curNode,curNodeData)
        if((impurity(curNodeData))!=0):
            curNode.best_feature(curNodeData, impurity)
            
            #for each child - find the data suitable to him and push the node and 
            #it's data to the NodeQueue and DataQueue respectively
            firstChild = DecisionNode()
            secondChild = DecisionNode()
            firstChildData, secondChildData = curNode.split_by_threshold(curNodeData)
            if(not make_a_split(curNodeData,firstChildData,secondChildData,chi_value)):
                curNode.isLeaf = True 
            else:
                NodeQueue.append(firstChild)
                DataQueue.append(firstChildData)
                NodeQueue.append(secondChild)
                DataQueue.append(secondChildData)
                #add the children to the current node  
                curNode.add_child(firstChild)
                curNode.add_child(secondChild)
        else:
            # if the impurity is 0 then the label is equal in each of the labels column - hence this is the prediction
            curNode.isLeaf = True   
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
    while((len(node.children))>0):
        if(instance[node.feature]>node.value):
            node = node.children[0]
        else:
            node = node.children[1]
    # the while loop will end with a node that has no kids.
    # meaning that it is a leaf and we know it's prediction        
    pred = node.prediction           
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_prediction(node, data):
    """
    calculate the prediction of the node based on the dataset

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the prediction is evaluated

    Output: the prediction of the node classication based on the given dataset (%).
    """
    unique, counts = np.unique(data, return_counts=True)
    d = dict(zip(unique, counts))
    if (d[0] > d[1]): return 0
    
    return 1



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
    sum = 0.0
    size = len(dataset)
    for row in dataset:
        pred = predict(node,row)
        if(pred == row[-1]): sum+=1

    accuracy = sum/size    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (accuracy*100)


def post_pruning(root, data):
 
    while (len(root.children)>0):
        bestAccuracy = -1
        bestParent = None
        possibleParents = possible_parents(root)
        # check the accuracy for each possible parent
        for currentParent in possibleParents:
            # trim parent children to check for the accuracy
            tempChildren = currentParent.children
            currentParent.children = []
            accuracy = calc_accuracy(root, data)
            if accuracy > bestAccuracy:
                bestParent = currentParent
                bestAccuracy = accuracy
            # return the trimmed children to the parent node 
            currentParent.children = tempChildren
            print(bestAccuracy)


        # trim the leafs of the best possible parent in the tree
        bestParent.children = []
        bestParent.isLeaf = True


def possible_parents(root):
    NodeQueue = [root]
    possibleParents = []
    while(len(NodeQueue)>0):
        curNode = NodeQueue.pop(0)
        if(curNode.children[0].isLeaf & curNode.children[1].isLeaf):
            possibleParents.append(curNode)   
        else:
            if (not curNode.children[0].isLeaf): NodeQueue.append(curNode.children[0])
            if (not curNode.children[1].isLeaf): NodeQueue.append(curNode.children[1])       

    return possibleParents            

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
