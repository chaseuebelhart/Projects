import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris_raw')
sys.path.insert(0, '../gym-tetris/gym_tetris')
import gym
import gym_tetris
import gym_tetris_raw
from randomAgent import *
import multiprocessing as mp
import operator
import time
import statistics as stat

def simulate(out):
    env = gym.make('Tetris-v1')
    env.reset()
    largest = 0
    agent = RandomAgent(env.action_space)
    while(True):
        action = agent.act()
        ob, reward, done, _ = env.step(action)
        if largest < env.env.game_state.piecesPlayed:
            largest = env.env.game_state.piecesPlayed
        if done:
            break

    out.put(largest)





class RandomSimulator:
    def __init__(self):
        self.what = 'A Random Simulator'


    def run(self, numProcesses):
        output = mp.Queue()
        processes = []

        # Create processes

        for processNum in range(numProcesses):
            # Select a subset of the total agents for this process to evaluate
            process = mp.Process(target=simulate, args=(output,))
            processes.append(process)

        # Start processes
        for process in processes:
            process.start()

        # Join processes
        for process in processes:
            process.join()

        # Combine results into a dictionary mapping agent id to fitness value
        return output

def Q_to_list(queue, runNum):
    i = 0
    list = []
    while i < runNum:
        i += 1
        list.append(queue.get())
    return list

def get_stats(array):
    return stat.mean(array) , stat.pstdev(array)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        runNum = int(sys.argv[1])
    else:
        runNum = 100
    numBatch = 1
    piece_count = []
    simulator = RandomSimulator()
    for i in range(numBatch):
        piece_count = piece_count + Q_to_list(simulator.run(runNum),runNum)

    #piece_count is a list of all the number of pieces played in a game for a graph
    out_mean, out_stdv = get_stats(piece_count)
    print('mean: ', out_mean, 'stdv: ', out_stdv)
