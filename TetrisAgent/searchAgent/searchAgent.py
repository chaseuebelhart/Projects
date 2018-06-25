import random
import string
import copy
from heuristicValueApproximator import HeuristicValueApproximator
from possibleStateGeneration import generateAllPossibleMoves, createStateMatrix
######
import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris')
import gym
import gym_tetris
import time

class SearchAgent:
    '''An agent that checks every possible resting location for the current
        piece, predicts which one is best using a value heuristic, and then
        returns the shortest series of actions to move the current piece to that
        location'''

    def __init__(self, envName, valueApproximator):
        self.envName = envName
        self.valueApproximator = valueApproximator
        self.id = self.generateRandomId()

    def generateRandomId(self, length=10):
        digits = string.ascii_uppercase + string.digits
        return ''.join(random.choice(digits) for i in range(length))

    def predictBestState(self, statesAndActions, debug=False):
        '''Returns the state with the highest predicted value according to the
            value heuristic'''
        bestStateAndActions = None
        highestValue = -1000000
        for stateAndActions in statesAndActions:
            state = stateAndActions['state']
            value = self.valueApproximator.predictValue(state)

            if value > highestValue:
                highestValue = value
                bestStateAndActions = stateAndActions

        if debug:
            print("Highest value", highestValue)
            print(bestStateAndActions['state'])
            print(bestStateAndActions['actions'])

        return bestStateAndActions

    def act(self, observation, debug=False):
        '''Returns a list of moves to manuever piece to desired spot according
            to the value heuristic'''
        currentPiece = observation[0]
        nextPiece = observation[1]
        state = createStateMatrix(observation[2])

        if currentPiece == None:
            # No-op
            return [0]

        if debug:
            print(currentPiece)

        statesAndActions = generateAllPossibleMoves(state, currentPiece)
        bestStateAndActions = self.predictBestState(statesAndActions, debug)

        if bestStateAndActions == None:
            # No-op
            return [0]

        bestActionSeries = bestStateAndActions['actions']
        return bestActionSeries

if __name__ == '__main__':
    # TESTING
    agent = SearchAgent("Tetris-v0", HeuristicValueApproximator())

    env = gym.make("Tetris-v0")
    done = False
    actions = []
    observation = env.reset()

    while not done:
        env.render()
        time.sleep(0.25)
        if len(actions) == 0:
            actions = agent.act(observation)
        observation, reward, done, _ = env.step(actions[0])
        del actions[0]

    print("Reward: " + str(reward))
