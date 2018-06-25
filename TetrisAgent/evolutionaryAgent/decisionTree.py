
class BinaryDecisionNode(object):
    def __init__(self, variable, trueValue, falseChild=None, trueChild=None):
        self.variable = variable
        self.trueValue = trueValue
        self.falseChild = falseChild
        self.trueChild = trueChild

    def setFalseChild(self, falseChild):
        self.falseChild = falseChild

    def setTrueChild(self, trueChild):
        self.trueChild = trueChild

    def clone(self):
        ''' Clones this node and its children recursively '''
        clonedNode = BinaryDecisionNode(self.variable, self.trueValue, None, None)

        clonedNode.setFalseChild(self.falseChild.clone())
        clonedNode.setTrueChild(self.trueChild.clone())

        return clonedNode

class LeafNode(object):
    def __init__(self, value):
        self.value = value

    def clone(self):
        return LeafNode(self.value)

    def __str__(self):
        return str(self.value)

class DecisionTree(object):
    def __init__(self, root):
        self.root = root

    def classify(self, features):
        # Pass features through decision tree and return leaf value
        reachedLeaf = False
        currentNode = self.root
        while not reachedLeaf:
            if features[currentNode.variable] == currentNode.trueValue:
                currentNode = currentNode.trueChild
            else:
                currentNode = currentNode.falseChild

            # Check if currentNode is a node or not
            if type(currentNode) is LeafNode:
                return currentNode.value

    def printSubTree(self, node, indentation):
        if type(node) is BinaryDecisionNode:
            result = ""
            # Print decision condition
            result += "%s%s = %s\n" % (indentation, str(node.variable), str(node.trueValue))
            # Recusively print left and right subtrees
            indentation += "  "
            result += self.printSubTree(node.trueChild, indentation)
            result += self.printSubTree(node.falseChild, indentation)
            return result
        else:
            # Print leaf value
            return "%s%s\n" % (indentation, str(node))

    def createCopy(self):
        return DecisionTree(self.root.clone())

    def __str__(self):
        # Print decision tree depth first
        return self.printSubTree(self.root, "")

if __name__ == '__main__':
    # Test example
    node1 = BinaryDecisionNode("Current Tetrimino", "O", None, LeafNode(1))
    node2 = BinaryDecisionNode("Current Tetrimino", "I", LeafNode(3), LeafNode(5))
    node1.setFalseChild(node2)

    tree = DecisionTree(node1)

    features = {"Current Tetrimino": "S"}
    #print(tree.classify(features))
    print("Original\n", tree)

    clonedTree = tree.createCopy()
    print("Copy\n", clonedTree)
