import copy
import math
import numpy as np

# shape constants - used for evaluating possible moves

SHAPES = {
  'I': np.rot90(np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])),
  'L': np.rot90(np.array([[1,0,0], [1,1,1], [0,0,0]])),
  'J': np.rot90(np.array([[0,0,1], [1,1,1], [0,0,0]])),
  'O': np.rot90(np.array([[1,1], [1,1]])),
  'Z': np.rot90(np.array([[0,1,1], [1,1,0], [0,0,0]])),
  'T': np.rot90(np.array([[0,1,0], [1,1,1], [0,0,0]])),
  'S': np.rot90(np.array([[1,1,0], [0,1,1], [0,0,0]]))
}
'''
SHAPES = {
  'I': [np.array([[0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0]]),
        np.array([[1,1,1,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]])],
  'L': [np.array([[0,1,0,0,0], [0,1,1,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,1,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,1,1,1,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,0,0], [0,0,1,0,0], [0,1,1,0,0], [0,0,0,0,0], [0,0,0,0,0]])],
  'J': [np.array([[0,0,0,1,0], [0,1,1,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,0,0], [0,0,1,0,0], [0,0,1,1,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,1,1,1,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]])],
  'O': [np.array([[0,1,1,0,0], [0,1,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]])],
  'Z': [np.array([[0,0,1,1,0], [0,1,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,0,0], [0,0,1,1,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0]])],
  'T': [np.array([[0,0,1,0,0], [0,1,1,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,0,0], [0,0,1,1,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,1,1,1,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]])],
  'S': [np.array([[0,1,1,0,0], [0,0,1,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]),
        np.array([[0,0,1,0,0], [0,1,1,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]])]
}
'''

SHAPE_ROTATIONS = {
  'I':2,
  'J':4,
  'L':4,
  'O':1,
  'S':2,
  'Z':2,
  'T':4,
  'Z':2
}

# given a gym-state, create a numpy array
def createStateMatrix(state):
  screen = state[2]
  matrix = np.zeros((len(state), len(state[0])))
  for r in range(len(state)):
    for i in range(len(state[0])):
      if state[r][i] == '.':
        matrix[r][i] = 0
      else:
        matrix[r][i] = 1
  return matrix

# return a rotated representation of a shape matrix

def rotateShapeMat(currentShape):
  new_shape = copy.copy(SHAPES[currentShape['shape']])
  for i in range(currentShape['rotation'] % SHAPE_ROTATIONS[currentShape['shape']]):
    new_shape = np.rot90(new_shape)
  return new_shape
'''
def rotateShapeMat(currentShape):
  new_shape = copy.copy(SHAPES[currentShape['shape']][currentShape['rotation']])
  return new_shape
'''

# shift the current position left
def moveLeft(state, shape, shape_mat):
  shape['x'] += 1
  # move down
  shape['y'] += 1
  if invalidPosition(state, shape, shape_mat):
    shape['x'] -= 1
    shape['y'] -= 1

def moveRight(state, shape, shape_mat):
  shape['x'] -= 1
  # move down
  shape['y'] += 1
  if invalidPosition(state, shape, shape_mat):
    shape['x'] += 1
    shape['y'] -= 1

# move the shape down
def moveDown(state, shape, shape_mat):
  shape['y'] += 1
  if invalidPosition(state, shape, shape_mat):
    shape['y'] -= 1
    return False
  return True

def drawShape(state, shape, shape_mat):
  new_state = copy.copy(state)
  for r in range(shape_mat.shape[0]):
    for c in range(shape_mat.shape[1]):
      if shape_mat[r][c] != 0:
        new_state[r + shape['x']][c + shape['y']] = 1
  return new_state

# determine if the current shape is in an invalid position
# for use with the generateAllPossibleMoves function
def invalidPosition(state, currentShape, shape_mat):
  for r in range(shape_mat.shape[0]):
    for c in range(shape_mat.shape[1]):
      if shape_mat[r][c] != 0:
        if (r + currentShape['x']) > (state.shape[0]-1) or (r + currentShape['x']) < 0:
          return True
        elif (c + currentShape['y']) > (state.shape[1]-1) or (c + currentShape['y']) < 0:
          return True
        elif state[r + currentShape['x']][c + currentShape['y']] != 0:
          return True
  return False

# generate all possible moves for where the current shape is
def generateAllPossibleMoves(state, currentShape):
  possible_moves = []
  shape = copy.copy(currentShape)
  # handle -1 offset inherent to engine
  shape['x'] += 1
  #if shape['shape'] == 'I':
#      shape['x'] += 1
  shape['rotation'] = shape['rotation'] % SHAPE_ROTATIONS[shape['shape']]

  # try all possible rotations
  for rotation in range(SHAPE_ROTATIONS[shape['shape']]):
    tried_x = []
    # 5 == rotate counter-clockwise action
    for d in range(-5, 6):
      #delta_r = (rotation + shape['rotation']) % SHAPE_ROTATIONS[shape['shape']]
      if rotation < 3:
        action_list = [2]*rotation
      else:
        action_list = [5]
      shape_move = copy.copy(shape)
      shape_move['rotation'] += rotation
      shape_mat = rotateShapeMat(shape_move)

      #if shape_move['shape'] == 'I' and shape_move['rotation'] % 2 == 1:
        #  shape_move['x'] += 1

      if d > 0:
        # move left
        for k in range(d):
          moveLeft(state, shape_move, shape_mat)
      elif d < 0:
        # move right
        for k in range(abs(d)):
          moveRight(state, shape_move, shape_mat)

      if shape_move['x'] not in tried_x or shape_move['x'] < 0 or shape_move['x'] > 9:
        # move the shape to the bottom of its current position
        while (moveDown(state, shape_move, shape_mat)):
          pass
        if invalidPosition(state, shape_move, shape_mat):
          continue

        # add on the actions of moving left/right
        if d > 0:
          if shape_move['shape'] == 'I' and shape_move['rotation'] % 2 == 1:
            d += 1
          if (shape_move['shape'] == 'Z' and shape_move['rotation'] % 2 == 1):
            d += 1
          action_list += [3]*d
        elif d < 0:
          if (shape_move['shape'] == 'Z' and shape_move['rotation'] % 2 == 1):
            d += 1
          elif shape_move['shape'] == 'I' and shape_move['rotation'] % 2 == 1:
            d += 1
          action_list += [1]*abs(d)
        else:
          if (shape_move['shape'] == 'Z' and shape_move['rotation'] % 2 == 1) or (shape_move['shape'] == 'I' and shape_move['rotation'] % 2 == 1):
            action_list.append(3)

        # drop the piece
        action_list.append(4)

        # add this state to the current possible states
        # first draw the current piece into the state
        new_state = drawShape(state, shape_move, shape_mat)
        move_dict = {'state':copy.copy(new_state), 'actions':copy.copy(action_list)}
        possible_moves.append(copy.copy(move_dict))
        tried_x.append(shape_move['x'])
  return possible_moves

# testing
if __name__ == '__main__':
  test_state = np.zeros((10, 20))
  test_state[0][19] = 1
  test_state[0][18] = 1
  test_state[1][19] = 1
  test_state[1][18] = 1
  test_state[2][19] = 1
  test_state[3][19] = 1
  test_state[3][18] = 1
  piece = {'shape':'S', 'rotation':0, 'x':3, 'y':0, 'color':2}
  possible_states = generateAllPossibleMoves(test_state, piece)

  print(possible_states)
