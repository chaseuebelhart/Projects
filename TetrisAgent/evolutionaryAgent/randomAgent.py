import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris')

import gym
import gym_tetris
import time

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    env = gym.make('Tetris-v0')

    agent = RandomAgent(env.action_space)

    episode_count = 10
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        for j in range(max_steps):
            env.render()
            time.sleep(5)
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print("")
            print(ob)
            if done:
                break
