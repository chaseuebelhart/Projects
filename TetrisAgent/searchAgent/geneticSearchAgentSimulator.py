import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris_raw')
sys.path.insert(0, '../gym-tetris/gym_tetris')
import gym
import gym_tetris
import gym_tetris_raw
import multiprocessing as mp
import operator
import time
import random
from heuristicValueApproximator import HeuristicValueApproximator
from searchAgent import SearchAgent

class GeneticSearchAgentSimulator:
    def __init__(self, envTrainName, envEvalName, generations, populationSize,
            samplesPerIndividual, mutationRate, mutationSD, numProcesses=8,
            debug=False):
        self.envTrainName = envTrainName
        self.envEvalName = envEvalName
        self.envEval = gym.make(envEvalName)
        self.generations = generations
        self.populationSize = populationSize
        self.samplesPerIndividual = samplesPerIndividual
        self.mutationRate = mutationRate
        self.mutationSD = mutationSD

        self.numProcesses = numProcesses
        self.debug = debug

        self.generateAgents()

    def simulate(self, agents, output):
        '''Simulate games for training and save the average rewards for each
            agent in output'''
        env = gym.make(envTrainName)
        rewards = {}

        for i in range(len(agents)):
            agent = agents[i]
            rewards[agent.id] = 0
            totalRewards = 0

            # Run multiple trials and average the rewards
            for iteration in range(samplesPerIndividual):
                done = False
                actions = []
                observation = env.reset()

                while not done:
                    if len(actions) == 0:
                        actions = agent.act(observation)
                    observation, reward, done, _ = env.step(actions[0])
                    totalRewards += reward
                    #del actions[0]
                    actions = actions[1:]

                rewards[agent.id] += totalRewards/samplesPerIndividual

        output.put(rewards)

    def visualizeGame(self, agent):
        '''Visualize a game for evaluation'''
        done = False
        actions = []
        observation = self.envEval.reset()
        totalRewards = 0
        score = 0
        pieces_played = 0
        lines_cleared = 0
        counter = 0

        while not done:
            self.envEval.render()

            if self.debug:
                input()
                print("Current piece:", observation[0])
            else:
                time.sleep(0.05)

            if len(actions) == 0:
                actions = agent.act(observation, self.debug)
            observation, reward, done, _ = self.envEval.step(actions[0])

            #print("Current piece:", observation[0])

            # metrics for paper recording
            if self.envEval.env.game_state.score > score:
                score = self.envEval.env.game_state.score
            if self.envEval.env.game_state.lines > lines_cleared:
                lines_cleared = self.envEval.env.game_state.lines
            if self.envEval.env.game_state.piecesPlayed > pieces_played:
                pieces_played = self.envEval.env.game_state.piecesPlayed

            totalRewards += reward
            del actions[0]

        print("Reward: " + str(totalRewards))
        print("Score: ", score)
        print("Lines cleared: ", lines_cleared)
        print("Pieces_played: ", pieces_played)

    def generateAgents(self):
        self.agents = {}

        for agent in range(self.populationSize):
            # Generate value approximator that is a random linear combination of
            # of heuristics
            valueApproximator = HeuristicValueApproximator()

            # Create a new SearchAgent and add it to the dict of agents
            agent = SearchAgent(self.envTrainName, valueApproximator)
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
            process = mp.Process(target=self.simulate, args=(agents, output))
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

    def reproduce(self, parent1, parent2, numChildren, mutationRate=0.2, mutationSD=0.1):
        children = []
        coeffecients1 = parent1.valueApproximator.coeffecients
        coeffecients2 = parent2.valueApproximator.coeffecients

        for childNum in range(numChildren):
            # Perform crossover between the two parents
            childCoeffecients = {}
            for key in coeffecients1.keys():
                if random.random() < 0.5:
                    childCoeffecients[key] = coeffecients1[key]
                else:
                    childCoeffecients[key] = coeffecients2[key]

            # Mutate coeffecients
            for key, value in childCoeffecients.items():
                if random.random() < mutationRate:
                    # Mutate coeffecient
                    coeffecient = random.gauss(value, mutationSD)
                    childCoeffecients[key] = coeffecient

            # Create value approximator and child objects
            valueApproximator = HeuristicValueApproximator(childCoeffecients)
            child = SearchAgent(self.envTrainName, valueApproximator)
            children.append(child)

        return children

    def updatePopulation(self, fitnessSorted):
        newAgents = {}

        # Select top 10% of agents for reproduction according to their fitness
        reproductiveAgents = []
        for idFitnessPair in fitnessSorted[:int(self.populationSize/10)]:
            agentId = idFitnessPair[0]
            reproductiveAgents.append(self.agents[agentId])

        # Randomly pair 2 agents for reproduction from the top 50%
        while len(reproductiveAgents)  >= 2:
            randomAgent1 = reproductiveAgents.pop(random.randrange(len(reproductiveAgents)))
            randomAgent2 = reproductiveAgents.pop(random.randrange(len(reproductiveAgents)))
            children = self.reproduce(randomAgent1, randomAgent2, 20,
                self.mutationRate, self.mutationSD)

            # Add children to newAgents
            for child in children:
                newAgents[child.id] = child

        # Update agents
        self.agents = newAgents

    def train(self):
        for generation in range(self.generations):
            print("\n--- Generation ", generation, " ---")

            # Print the population ids
            #print("\nPopulation: ", self.agents.keys())
            #print("Population size:", len(self.agents))

            # Evaluate fitness of individuals
            fitness = self.evaluateFitness(self.numProcesses)

            # Sort the fitness values in decreasing order
            fitnessSorted = sorted(fitness.items(), key=operator.itemgetter(1), reverse=True)
            #print("\nfitnessSorted: ", fitnessSorted)

            # Fittest individual
            fittestId = fitnessSorted[0][0]
            fittestAgent = self.agents[fittestId]

            # Print population statistics
            print("Highest fitness: ", fitnessSorted[0])
            print("90th percentile fitness", fitnessSorted[int(self.populationSize/10)])
            print("75th  percentile fitness: ", fitnessSorted[int(self.populationSize/4)])
            print("50th  percentile fitness: ", fitnessSorted[int(self.populationSize/2)])

            # Visualize fittest individual
            self.visualizeGame(fittestAgent)

            # Print best coeffecients
            print("Best coeffecients:", fittestAgent.valueApproximator.coeffecients)

            # Update the population
            self.updatePopulation(fitnessSorted)

if __name__ == '__main__':
    envTrainName = 'Tetris-v1'
    envEvalName = 'Tetris-v0'
    generations = 500
    populationSize = 160
    samplesPerIndividual = 3
    mutationRate = 0.3
    mutationSD = 0.15
    numProcesses = 8
    debug = False

    simulator = GeneticSearchAgentSimulator(envTrainName, envEvalName, generations,
        populationSize, samplesPerIndividual, mutationRate, mutationSD,
        numProcesses, debug)
    simulator.train()
