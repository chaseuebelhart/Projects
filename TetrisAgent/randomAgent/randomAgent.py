import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris')

import gym
import gym_tetris
import time

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':
    env = gym.make('Tetris-v0')

    agent = RandomAgent(env.action_space)

    episode_count = 10
    max_steps = 200
    largest = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        for j in range(max_steps):
            #env.render()
            action = agent.act()
            ob, reward, done, _ = env.step(action)
            if largest < env.env.game_state.piecesPlayed:
                largest = env.env.game_state.piecesPlayed
            if done:
                break
        print(largest)
        largest = 0
