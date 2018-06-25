import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris_raw')
sys.path.insert(0, '../gym-tetris/gym_tetris')
import gym
import gym_tetris
import gym_tetris_raw
from evolutionaryAgent import *
from EvolNN import NNAgent
import multiprocessing as mp
import operator
import time
from featureFunctions import num_holes

def computeReward(observation, rowsCleared):
    # This might not be right
    numFilledBlocks = 0
    blocks = observation[2]
    for column in blocks:
        for square in column:
            if square != ".":
                numFilledBlocks += 1
    return numFilledBlocks + rowsCleared


def simulateStats(envName, agent, maxSteps, iterations):
    env = gym.make(envName).unwrapped
    rewards = {}
    stats = {"pieces":0, "score":0, "lines":0}
    rewards[agent.id] = 0
    print(type(env))
    print(type(env.game_state))
    for iteration in range(iterations):
        done = False
        positiveRewards = 0
        observation = env.reset()
        for step in range(maxSteps):
            action = agent.act(observation)
            lastObservation = observation
            		
            pi = env.game_state.piecesPlayed/iterations
            sc = env.game_state.score/iterations
            li = env.game_state.lines/iterations
            observation, reward, done, _ = env.step(action)

            if reward > 0:
                positiveRewards += reward

            if done:
                stats["pieces"] += pi
                stats["score"] += sc
                stats["lines"] += li
                break
    return stats

def simulate(envName, agents, maxSteps, iterations, output):
    env = gym.make(envName)
    rewards = {}

    for i in range(len(agents)):
        agent = agents[i]
        rewards[agent.id] = 0
        for iteration in range(iterations):
            done = False
            positiveRewards = 0
            observation = env.reset()
            for step in range(maxSteps):
                action = agent.act(observation)
                lastObservation = observation
                observation, reward, done, _ = env.step(action)

                if reward > 0:
                    positiveRewards += reward

                if done:
                    reward = computeReward(lastObservation, positiveRewards)
                    #print("Reward: " + str(reward))
                    rewards[agent.id] += reward/iterations
                    break

    output.put(rewards)

def visualizeTetrisGame(agent, env):
    rewards = {}
    done = False
    positiveRewards = 0

    observation = env.reset()
    for step in range(maxSteps):
        env.render()
        time.sleep(0.04)
        action = agent.act(observation)

        board = copy.deepcopy(observation[2])
        for col in range(0, len(board)):
            for row in range(0, len(board[col])):
                if board[col][row] == ".":
                    board[col][row] = False
                else:
                    board[col][row] = True
        # print(num_holes(board))
        print(action)
        print(agent.classify(agent.observationToFeatures(observation)))
        lastObservation = observation
        observation, reward, done, _ = env.step(action)

        if reward > 0:
            positiveRewards += reward

        if done:
            reward = computeReward(lastObservation, positiveRewards)
            print("Positive rewards: " + str(positiveRewards))
            print("Reward: " + str(reward))
            break

class EvolutionSimulator(object):
    def __init__(self, envName, populationSize, generations, iterations, maxSteps, mutationRate, numProcesses=8):
        self.envName = envName
        self.populationSize = populationSize
        self.generations = generations
        self.iterations = iterations
        self.maxSteps = maxSteps
        self.mutationRate = mutationRate
        self.numProcesses = numProcesses
        self.evalEnvironment = gym.make("Tetris-v0")
        self.bestAgent = None
        self.generateAgents()

    def generateAgents(self):
        env = gym.make(envName)
        self.agents = {}
        for agent in range(populationSize):
            # Create a new EvolutionaryAgent and add it to the dict of agents
            agent = NNAgent(env.action_space)
            self.agents[agent.id] = agent

    def evaluateFitness(self, numProcesses):
        populationPerProcess = int(self.populationSize / numProcesses)
        output = mp.Queue()
        processes = []

        # Create processes
        allAgents = list(self.agents.values())
        for processNum in range(numProcesses):
            # Select a subset of the total agents for this process to evaluate
            agents = allAgents[processNum*populationPerProcess : (processNum+1)*populationPerProcess]
            process = mp.Process(target=simulate, args=(self.envName, agents, self.maxSteps, self.iterations, output))
            processes.append(process)

        # Start processes
        for process in processes:
            process.start()

        # Join processes
        for process in processes:
            process.join()

        # Combine results into a dictionary mapping agent id to fitness value
        fitness = {}
        for process in processes:
            fitnessDict = output.get()
            for k, v in fitnessDict.items():
                fitness[k] = v

        return fitness

    def updatePopulation(self, fitnessSorted):
        newAgents = {}

        # Add best current agent to pool of new agents
        bestAgentId = fitnessSorted[0][0]
        newAgents[bestAgentId] = self.agents[bestAgentId]
        self.bestAgent = self.agents[bestAgentId]
        # Select 50% of agents for reproduction according to their fitness
        reproductiveAgents = []
        for idFitnessPair in fitnessSorted[:int(self.populationSize/2)]:
            agentId = idFitnessPair[0]
            reproductiveAgents.append(self.agents[agentId])

        # Perform crossover and mutation to create 2 children per agent selected
        for agent in reproductiveAgents:
            for childNum in range(2):
                child = agent.reproduce(self.mutationRate)
                newAgents[child.id] = child

        # Update agents
        self.agents = newAgents

    def run(self):
        for generation in range(generations):
            print("\n--- Generation ", generation, " ---")

            # Print the population ids
            #print("\nPopulation: ", self.agents.keys())

            # Evaluate fitness of individuals
            fitness = self.evaluateFitness(self.numProcesses)

            # Sort the fitness values in decreasing order
            fitnessSorted = sorted(fitness.items(), key=operator.itemgetter(1), reverse=True)
            #print("\nfitnessSorted: ", fitnessSorted)

            # Print the decision tree of the fittest agent
            fittestId = fitnessSorted[0][0]
            fittestAgent = self.agents[fittestId]
            print(fittestAgent.clf)

            # Print population statistics
            print("Highest fitness: ", fitnessSorted[0])
            print("Median fitness: ", fitnessSorted[int(self.populationSize/2)])

            visualizeTetrisGame(fittestAgent, self.evalEnvironment)

            # Update the population
            self.updatePopulation(fitnessSorted)
        return self.bestAgent

if __name__ == '__main__':
    envName = 'Tetris-v1'
    generations = 20
    iterations = 3
    populationSize = 800
    maxSteps = 10000
    mutationRate = 0.1

    simulator = EvolutionSimulator(envName, populationSize, generations, iterations, maxSteps, mutationRate)
    agent = simulator.run()
    stats = simulateStats('Tetris-v0', agent, maxSteps, 100)
    print(stats)
    
