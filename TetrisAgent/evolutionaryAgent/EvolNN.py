import random
import copy
import string
import time
from featureFunctions import *

FEATURES = [
    # "Current Shape",
    # "Next Shape",
    "Current X",
    "Current Y",
    # "Current Rotation",
    # "Max Height",
    "Drop Creates Cave",
    "totalHeight",
    "maxHoleHeight",
    "numHoles",
    # "playableRow",
    "numHoleRows",
    "numBlockades",
    "numHoleCols",
    # "clearedLines",
    "highestCol",
    "bump"
]

SHAPES = {
    "S": 0,
    "I": 1,
    "L": 2,
    "J": 3,
    "Z": 4,
    "O": 5
}

ACTIONS = {
#            0: "No-Op"
            1: "Translate right",
            2: "Rotate counter-clockwise",
            3: "Translate left",
            4: "Drop",
            5: "Rotate clockwise"
#            6: "Random"
            }

class NNAgent(object):
    def __init__(self, action_space, clf=None):
        self.id = self.generateRandomId()
        self.action_space = action_space

        if clf == None:
            # Randomly generate decision tree
            self.clf = self.randomlyGenerateCLF()
            #print(self.decisionTree)
        else:
            self.clf = clf

    def generateRandomId(self, length=10):
        digits = string.ascii_uppercase + string.digits
        return ''.join(random.choice(digits) for i in range(length))

    def observationToFeatures(self, observation):
        fallingPiece = observation[0]
        board = copy.deepcopy(observation[2])
        for col in range(0, len(board)):
            for row in range(0, len(board[col])):
                if board[col][row] == ".":
                    board[col][row] = False
                else:
                    board[col][row] = True
        if not fallingPiece:
            fallingPiece = {"x": 5, "y": 20}
        # features = {
        #             # "Current Shape": fallingPiece["shape"],
        #             # "Current X": fallingPiece["x"],
        #             # # "Current Y": fallingPiece["y"],
        #             # "Current Rotation": fallingPiece["rotation"],
        #             # "Next Shape": observation[1]["shape"],
        #             # "Max Height": observation[3],
        #             # "Drop Creates Cave": observation[4],
        #             "totalHeight":          total_height(board),
        #             "maxHoleHeight":        max_hole_height(board),
        #             # "numHoles":             num_holes(board),
        #             # "playableRow":          playable_row(board),
        #             # "clearedLines":         observation[5],
        #             # "numHoleRows":          num_hole_rows(board),
        #             # "numHoleCols":          num_hole_cols(board),
        #             "numBlockades":         num_blockades(board),
        #             "highestCol":           highest_col(board),
        #             "bump":                 bumpiness(board)
        #             }
        features = [
            # SHAPES[fallingPiece["shape"]],
            int(fallingPiece["x"])/10,
            int(fallingPiece["y"])/20,
            int(observation[4]),
            total_height(board)/200 ,
            max_hole_height(board)/20,
            num_blockades(board)/190,
            num_hole_cols(board)/10,
            num_hole_rows(board)/20,
            max_hole_height(board)/20,
            highest_col(board)/10,
            bumpiness(board)/170
       ]
        return features

    def classify(self, features):
        output = [0 for i in range(len(list(ACTIONS.keys())))]
        for action in range(len(self.clf)):
            for feature in range(len(self.clf[action])):
                output[action]+= self.clf[action][feature] * features[feature]

        return output.index(max(output)) + 1

    def act(self, observation):
        try:
            # Convert observation into a dictionary of features
            features = self.observationToFeatures(observation)
            action = self.classify(features)
            if action == 6:
                 action = random.randrange(1,4)
            return action
        except:
            # Error occured. Perform No-op
            return 0

    def reproduce(self, mutationRate):
        ''' Returns a new child by performing crossover and mutation'''
        # Create a copy of the decision tree
        copyCLF = copy.deepcopy(self.clf)

        for action in range(len(copyCLF)):
            for feature in range(len(copyCLF[action])):
                rng = random.random()
                if rng<mutationRate:
                    copyCLF[action][feature] = random.random()

        # Perform mutations to the decision tree copy
        # self.mutateNodeAndChildren(decisionTreeCopy.root, mutationRate)

        # Create new agent with created decision tree
        newAgent = NNAgent(self.action_space, copyCLF)

        return newAgent

    def randomlyGenerateCLF(self):
        # Create unattached nodes and put them in a list
        return [[random.random() for feature in range(len(FEATURES))] for action in range(len(list(ACTIONS.keys())))]


    def handDesignedDecisionTree(self):
        pass
