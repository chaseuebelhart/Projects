import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris_raw')
sys.path.insert(0, '../gym-tetris/gym_tetris')
import gym
import gym_tetris
import gym_tetris_raw
from evolutionaryAgent import *
import multiprocessing as mp
import operator
import time

def computeReward(observation, rowsCleared):
    # This might not be right
    numFilledBlocks = 0
    blocks = observation[2]
    for column in blocks:
        for square in column:
            if square != ".":
                numFilledBlocks += 1
    return numFilledBlocks + rowsCleared

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
        time.sleep(0.01)
        action = agent.act(observation)
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
    def __init__(self, envName, populationSize, generations, iterations, maxSteps,
        mutationRate, initialDecisionTreeSize, numProcesses=8):
        self.envName = envName
        self.populationSize = populationSize
        self.generations = generations
        self.iterations = iterations
        self.maxSteps = maxSteps
        self.mutationRate = mutationRate
        self.numProcesses = numProcesses
        self.evalEnvironment = gym.make("Tetris-v0")
        self.initialDecisionTreeSize = initialDecisionTreeSize

        self.generateAgents()

    def generateAgents(self):
        env = gym.make(envName)
        self.agents = {}
        for agent in range(populationSize):
            # Create a new EvolutionaryAgent and add it to the dict of agents
            agent = EvolutionaryAgent(env.action_space,
                initialDecisionTreeSize=self.initialDecisionTreeSize)
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
            print(fittestAgent.decisionTree)

            # Print population statistics
            print("Highest fitness: ", fitnessSorted[0])
            print("Median fitness: ", fitnessSorted[int(self.populationSize/2)])

            visualizeTetrisGame(fittestAgent, self.evalEnvironment)

            # Update the population
            self.updatePopulation(fitnessSorted)

if __name__ == '__main__':
    envName = 'Tetris-v1'
    generations = 200
    iterations = 3
    populationSize = 800
    maxSteps = 10000
    mutationRate = 0.1
    initialDecisionTreeSize = 4

    simulator = EvolutionSimulator(envName, populationSize, generations, iterations, \
        maxSteps, mutationRate, initialDecisionTreeSize)
    simulator.run()
