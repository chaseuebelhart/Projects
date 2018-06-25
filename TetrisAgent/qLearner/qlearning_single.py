import sys
import time
import os
import numpy as np
sys.path.insert(0, '../gym-tetris/gym_tetris')
#sys.path.insert(0, '../gym-tetris/gym_tetris_raw')
import gym
import gym_tetris
#import gym_tetris_raw
from itertools import count
import random
import math
from collections import namedtuple
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T

env = gym.make('Tetris-v0').unwrapped

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor



class QNet(nn.Module):
 
  def __init__(self):
    # network architecture inspired by Human-level control through deep 
    # reinforcement learning (Mnih et. al)
    super(QNet, self).__init__()
    # one input channel, 32 output channels, 5x5 conv filter
    self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
    self.conv2 = nn.Conv2d(32, 32, 3, padding = 1)
    self.conv3 = nn.Conv2d(32, 32, 3, padding = 1)
    self.conv4 = nn.Conv2d(32, 64, (1, 20))
    self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
    self.conv6 = nn.Conv2d(128, 128, 1)
    self.conv7 = nn.Conv2d(128, 128, 3, padding = 1)
    # fully connected layers
    self.fc1 = nn.Linear(10*128, 128)
    self.fc2 = nn.Linear(128, 512)
    # output layer, 6 possible moves
    self.fc3 = nn.Linear(512, 6)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))
    x = x.view(-1, 10*128)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return x

# standard replay memory class and transition tuple 
# obtained from PyTorch website
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# tunable hyperparameters
BATCH_SIZE = 1200
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 100

steps_done = 0

model = QNet()
model_cache = QNet()
model_cache.load_state_dict(model.state_dict())
model_cache.eval()

loss = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=.001)
memory = ReplayMemory(20000)

# grouped actions
'''
ACTION_MAP = {
  0: [0],
  1: [1, 4],
  2: [1, 1, 4],
  3: [1, 1, 1, 4],
  4: [1, 1, 1, 1, 4],
  5: [1, 1, 1, 1, 1, 4],
  6: [2, 1, 4],
  7: [2, 1, 1, 4],
  8: [2, 1, 1, 1, 4],
  9: [2, 1, 1, 1, 1, 4],
  10: [2, 1, 1, 1, 1, 1, 4],
  11: [2, 2, 1, 4],
  12: [2, 2, 1, 1, 4],
  13: [2, 2, 1, 1, 1, 4],
  14: [2, 2, 1, 1, 1, 1, 4],
  15: [2, 2, 1, 1, 1, 1, 1, 4],
  11: [5, 1, 4],
  12: [5, 1, 1, 4],
  13: [5, 1, 1, 1, 4],
  14: [5, 1, 1, 1, 1, 4],
  15: [5, 1, 1, 1, 1, 1, 4],
  16: [3, 4],
  17: [3, 3, 4],
  18: [3, 3, 3, 4],
  19: [3, 3, 3, 3, 4],
  20: [3, 3, 3, 3, 3, 4],
  21: [2, 3, 4],
  22: [2, 3, 3, 4],
  23: [2, 3, 3, 3, 4],
  24: [2, 3, 3, 3, 3, 4],
  25: [2, 3, 3, 3, 3, 3, 4],
  26: [2, 2, 3, 4],
  27: [2, 2, 3, 3, 4],
  28: [2, 2, 3, 3, 3, 4],
  29: [2, 2, 3, 3, 3, 3, 4],
  30: [2, 2, 3, 3, 3, 3, 3, 4],
  31: [5, 3, 4],
  32: [5, 3, 3, 4],
  33: [5, 3, 3, 3, 4],
  34: [5, 3, 3, 3, 3, 4],
  35: [5, 3, 3, 3, 3, 3, 4],
  36: [4],
  37: [2, 4],
  38: [2, 2, 4],
  39: [5, 4]
}
'''
'''
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

#def try_possible_states(state)
def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)
'''

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return FloatTensor([[random.randrange(6)]])
        #return FloatTensor([[random.randrange(40)]])

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
    next_state_values[non_final_mask] = model_cache(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    if len(loss.data)>0 : return loss.data[0] 
    else : return loss

def compute_state_heuristic(state):
    height = 0
    num_holes = 0
    bumpiness = 0
    prev_height = 0
    max_height = 0
    for i in range(10):
        # fast way of checking the number of holes
        num_holes += np.where(np.sum(state[i][0:-1] - state[i][1:]) > 0)[0].size
        # special case where there is a hole at the bottom of this column
        if state[i][-1] == 0 and np.sum(state[i]) > 0:
            num_holes += 1
        # height aggregation
        spots = np.where(state[i] == 1)
        if spots[0].size > 0:
            this_height = 10 - spots[0][0]
            if this_height > max_height:
                max_height = this_height
            if i > 0:
                bumpiness += np.abs(this_height - prev_height)
            height += this_height
            prev_height = this_height
    return -0.36*num_holes + -0.25*bumpiness - 0.6*max_height
        
          
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    try: # If these fail, its loading a supervised model
        optimizer.load_state_dict(checkpoint['optimizer'])
        memory = checkpoint['memory']
    except Exception as e:
        pass
    # Low chance of random action
    #steps_done = 10 * EPS_DECAY

    return checkpoint['epoch']


def extract_state_tensor(state):
    screen = state[2]
    val_screen = np.zeros((len(screen), len(screen[0])))
    for r in range(len(screen)):
        for i in range(len(screen[0])):
            if screen[r][i] == '.':
                val_screen[r][i] = 0
            else:
                val_screen[r][i] = 1
    partial_fitness = compute_state_heuristic(val_screen)
    
    # add in current piece configuration
    piece_details = state[0]
    piece_string = 'SZIOJLT'
    if piece_details:
        val_screen[piece_details['x']][piece_details['y']] = \
          10*piece_string.find(piece_details['shape']) + piece_details['rotation']

    return partial_fitness, FloatTensor(val_screen[None, None, :, :])

current_episode = 0
#num_episodes = 50
#for i_episode in range(num_episodes):

if len(sys.argv) > 1 and sys.argv[1] == 'resume':
    if len(sys.argv) > 2:
        CHECKPOINT_FILE = sys.argv[2]
    if os.path.isfile(CHECKPOINT_FILE):
        print("=> loading checkpoint '{}'".format(CHECKPOINT_FILE))
        start_epoch = load_checkpoint(CHECKPOINT_FILE)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(CHECKPOINT_FILE, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(CHECKPOINT_FILE))

scores = np.zeros(20)
pieces_played = np.zeros(20)
lines_cleared = np.zeros(20)
while True:
    # Initialize the environment and state
    env.reset()

    state, _, _, _ = env.step(0)
    partial_reward, state = extract_state_tensor(state)
    prev_reward = partial_reward
    current_score = 0
    current_lines = 0
    num_pieces_played = 0
    for t in count():
        # Select and perform an action
        env.render()
        action = select_action(state).type(LongTensor)
        
        #for a in ACTION_MAP[action[0, 0]]:
        next_state, reward, done, _ = env.step(action[0,0])
        partial_reward, next_state = extract_state_tensor(next_state)
        if current_episode > 0:
            reward += partial_reward
            reward = reward - prev_reward
            prev_reward = reward

        if done:
            reward -= 20 
        reward = FloatTensor([reward])
        
        if env.game_state.score > current_score:
            current_score = env.game_state.score
        if env.game_state.lines > current_lines:
            current_lines = env.game_state.lines
        if env.game_state.piecesPlayed > num_pieces_played:
            num_pieces_played = env.game_state.piecesPlayed
        # Store the transition in memory
        
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        if done:
            scores[current_episode%20] = current_score
            pieces_played[current_episode % 20] = num_pieces_played
            lines_cleared[current_episode % 20] = current_lines
            if current_episode % 20 == 0:
                  optimize_model()
            if current_episode % 20 == 0:
                  print(current_episode, "score: ", np.mean(scores), "pp: ", np.mean(pieces_played), 
                        "lc: ", np.mean(lines_cleared))
                  scores = np.zeros(20) 
                  lines_cleared = np.zeros(20)
                  pieces_played = np.zeros(20)
                  save_checkpoint({
                      'epoch' : current_episode,
                      'state_dict' : model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                      'memory' : memory
                      })
                              
            break

    if current_episode % TARGET_UPDATE == 0:
        model_cache.load_state_dict(model.state_dict())

    current_episode += 1

