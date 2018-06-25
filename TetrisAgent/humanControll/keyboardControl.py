import sys, os.path
sys.path.insert(0, '../gym-tetris/gym_tetris')

import pygame
import gym
import gym_tetris
import time

output = './output-metrics.txt'

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    env = gym.make('Tetris-v0')

    episode_count = 1
    max_steps = 200000000
    reward = 0
    done = False
    action = 0

    f = open(output, 'a')
    for i in range(episode_count):
        ob = env.reset()
        for j in range(max_steps):
            env.render()
            time.sleep(0.5)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:#move left
                        action = 3
                    elif event.key == pygame.K_RIGHT:#move right
                        action = 1
                    elif event.key == pygame.K_UP:#rotate one way
                        action = 2
                    elif event.key == pygame.K_DOWN:#rotate the other way
                        action = 5
                    elif event.key == pygame.K_SPACE:#drop the piece
                        action = 4
                    else:                           #no op
                        action = 0


            linesCleared = env.env.game_state.lines
            score = env.env.game_state.score
            piecesPlayed = env.env.game_state.piecesPlayed
            ob, reward, done, _ = env.step(action)
            action = 0
            if done:
                out = 'Game '+str(i+1)+' : Score = '+str(score)+ ' Lines Cleared = '+str(linesCleared)+ ' Pieces Played = '+ str(piecesPlayed)
                f.write(out)
                #print('Game ',i+1,' : Score = ',score, ' Lines Cleared = ',linesCleared, 'Pieces Played = ', piecesPlayed)
                print('\n\n-------------------------Results-------------------------')
                print(out)
                print('---------------------------------------------------------\n\n')
                break
