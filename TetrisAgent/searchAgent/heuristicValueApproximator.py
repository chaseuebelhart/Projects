import sys
sys.path.insert(0, '../gym-tetris/gym_tetris_raw')
sys.path.insert(0, '../gym-tetris/gym_tetris')
import random
from valueApproximator import ValueApproximator
from heuristicFunctions import *

class HeuristicValueApproximator(ValueApproximator):
    '''Approximates the value of a state using a linear combination of
        heuristics such as number of holes, bumpiness, etc'''

    def __init__(self, coeffecients=None):
        self.heuristics = ["numHoles", "bumpiness", "maxHeight",
            "numCompletedLines", "avgHeight", "existenceOfIValley"]

        if coeffecients == None:
            # Randomly initialize coeffecients between -1 and 1
            self.coeffecients = {}
            for heuristic in self.heuristics:
                self.coeffecients[heuristic] = random.uniform(-0.2, 0.2)
        else:
            self.coeffecients = coeffecients

    def predictValue(self, state):
        value = 0

        maxHeight, avgHeight = heights(state)

        value += self.coeffecients["numHoles"] * numHoles(state)
        value += self.coeffecients["bumpiness"] * bumpiness(state)
        value += self.coeffecients["maxHeight"] * maxHeight
        value += self.coeffecients["avgHeight"] * avgHeight
        value += self.coeffecients["numCompletedLines"] * numCompletedLines(state)
        value += self.coeffecients["existenceOfIValley"] * existanceOfIValley(state, maxHeight)

        return value
