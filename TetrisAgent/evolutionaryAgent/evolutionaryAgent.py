from decisionTree import *
import time
import random
import string
import copy

# Format: "Feature name": (Measurement type, Possible values)
FEATURES = {
            "Current Shape": ("nominal", ("S", "Z", "L", "J", "O", "I")),
            "Next Shape": ("nominal", ("S", "Z", "L", "J", "O", "I")),
            "Current X": ("ratio", (0,1,2,3,4,5,6,7,8,9)),
            "Current Y": ("ratio", (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)),
            "Current Rotation": ("nominal", (0,1,2,3)),
            "Max Height": ("ratio", (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)),
            "Drop Creates Cave": ("nominal", (True, False))
            }

ACTIONS = {
            0: "No-Op",
            1: "Translate right",
            2: "Rotate counter-clockwise",
            3: "Translate left",
            4: "Drop",
            5: "Rotate clockwise",
            6: "Random"
            }

class EvolutionaryAgent(object):
    def __init__(self, action_space, decisionTree=None, initialDecisionTreeSize=1):
        self.id = self.generateRandomId()
        self.action_space = action_space

        if decisionTree == None:
            # Randomly generate decision tree
            self.decisionTree = self.randomlyGenerateDecisionTree(initialDecisionTreeSize)
            #print(self.decisionTree)
        else:
            self.decisionTree = decisionTree

    def generateRandomId(self, length=10):
        digits = string.ascii_uppercase + string.digits
        return ''.join(random.choice(digits) for i in range(length))

    def generateRandomLeaf(self):
        randomAction = random.choice(list(ACTIONS.keys()))
        return LeafNode(randomAction)

    def generateRandomDecisionNode(self, falseChild, trueChild):
        randomVariable = random.choice(list(FEATURES.keys()))
        randomValue = random.choice(FEATURES[randomVariable][1])
        return BinaryDecisionNode(randomVariable, randomValue, falseChild, trueChild)

    def observationToFeatures(self, observation):
        fallingPiece = observation[0]

        features = {
                    "Current Shape": fallingPiece["shape"],
                    "Current X": fallingPiece["x"],
                    "Current Y": fallingPiece["y"],
                    "Current Rotation": fallingPiece["rotation"],
                    "Next Shape": observation[1]["shape"],
                    "Max Height": observation[3],
                    "Drop Creates Cave": observation[4]
                    }
        return features

    def act(self, observation):
        try:
            # Convert observation into a dictionary of features
            features = self.observationToFeatures(observation)
            action = self.decisionTree.classify(features)
            if action == 6:
                 action = random.randrange(1,4)
            return action
        except:
            # Error occured. Perform No-op
            return 0

    def mutateNodeAndChildren(self, rootNode, mutationRate):
        ''' Traverses decision tree in a depth first manner and randomly
        performs mutations '''

        if random.random() < mutationRate:
            if type(rootNode) is BinaryDecisionNode:
                mutationNum = random.randrange(0,3)

                if mutationNum == 0:
                    # Change decision variable and decision value
                    randomVariable = random.choice(list(FEATURES.keys()))
                    randomValue = random.choice(FEATURES[randomVariable][1])
                    rootNode.variable = randomVariable
                    rootNode.trueValue = randomValue

                elif mutationNum == 1:
                    # Add new decison node as a child
                    childNum = random.randrange(0,2)
                    if childNum == 0:
                        child = rootNode.falseChild
                    else:
                        child = rootNode.trueChild

                    # Randomly generate new nodes
                    newLeafNode = self.generateRandomLeaf()
                    newDecisionNode = self.generateRandomDecisionNode(child, newLeafNode)

                    # Update rootNode children
                    if childNum == 0:
                        rootNode.falseChild = newDecisionNode
                    else:
                        rootNode.trueChild = newDecisionNode

                elif mutationNum == 2:
                    # Delete a child decision node if possible
                    if type(rootNode.falseChild) is BinaryDecisionNode:
                        # Delete false child
                        rootNode.falseChild = rootNode.falseChild.falseChild
                    elif type(rootNode.trueChild) is BinaryDecisionNode:
                        # Delete true child
                        rootNode.trueChild = rootNode.trueChild.falseChild

            elif type(rootNode) is LeafNode:
                # Perform leaf node mutations on rootNode
                # Mutate value
                rootNode.value = random.choice(list(ACTIONS.keys()))

        # Recursively mutate children
        if type(rootNode) is BinaryDecisionNode:
            self.mutateNodeAndChildren(rootNode.falseChild, mutationRate)
            self.mutateNodeAndChildren(rootNode.trueChild, mutationRate)

    def reproduce(self, mutationRate):
        ''' Returns a new child by performing crossover and mutation'''
        # Create a copy of the decision tree
        decisionTreeCopy = self.decisionTree.createCopy()

        # Perform mutations to the decision tree copy
        self.mutateNodeAndChildren(decisionTreeCopy.root, mutationRate)

        # Create new agent with created decision tree
        newAgent = EvolutionaryAgent(self.action_space, decisionTreeCopy)

        return newAgent

    def randomlyGenerateDecisionTree(self, numNodes):
        # Create unattached nodes and put them in a list
        nodes = []
        while len(nodes) < numNodes:
            randomVariable = random.choice(list(FEATURES.keys()))
            randomValue = random.choice(FEATURES[randomVariable][1])
            falseAction = self.generateRandomLeaf()
            trueAction = self.generateRandomLeaf()
            node = BinaryDecisionNode(randomVariable, randomValue, falseAction, trueAction)
            nodes.append(node)

        # Create decision tree from the list of nodes
        tree = DecisionTree(nodes[0])
        treeNodes = [nodes[0]]
        del nodes[0]
        while len(nodes) > 0:
            randomTreeNode = random.choice(treeNodes)
            if random.random() < 0.5 and type(randomTreeNode.falseChild) is not BinaryDecisionNode:
                randomTreeNode.falseChild = nodes[0]
                treeNodes.append(nodes[0])
                del nodes[0]
            elif random.random() < 0.5 and type(randomTreeNode.trueChild) is not BinaryDecisionNode:
                randomTreeNode.trueChild = nodes[0]
                treeNodes.append(nodes[0])
                del nodes[0]

        return tree

    def handDesignedDecisionTree(self):
        pass
