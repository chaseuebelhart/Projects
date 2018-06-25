from statistics import stdev
import numpy as np

def numCompletedLines(state):
    '''Takes in a binary 2d numpy array of the board and returns the number of
        completed rows'''
    boardWidth = 10
    numCompletedLines = 0
    rowSums = np.sum(state, axis=0)

    # Count the number of rows with a sum equal to boardWidth
    for rowSum in rowSums:
        if rowSum == boardWidth:
            numCompletedLines += 1

    return numCompletedLines

def numHoles(state):
    '''Takes in a binary 2d numpy array of the board and returns the number of
        holes'''
    numHoles = 0

    for i in range(10):
        for j in range(19):
            if state[i][j] == 1:
                # Count the number of zeros below this 1
                numZeros = 20 - j - np.sum(state[i][j:])
                numHoles += numZeros
                break

    return numHoles

def heights(state):
    maxHeight = 0
    avgHeight = 0

    rowSums = np.sum(state, axis=0)

    # Compute max height
    for y in range(len(rowSums)):
        if rowSums[y] > 0:
            maxHeight = 20 - y
            break

    # Compute avgHeight
    for y in range(len(rowSums)):
        avgHeight += (20 - y) * rowSums[y]
    avgHeight /= 20

    return maxHeight, avgHeight

def bumpiness(state):
    bumpiness = 0
    prevHeight = 0

    for i in range(10):
        # height aggregation
        spots = np.where(state[i] == 1)
        if spots[0].size > 0:
            thisHeight = 10 - spots[0][0]
            if i > 0:
                bumpiness += np.abs(thisHeight - prevHeight)
            prevHeight = thisHeight

    return bumpiness

def existanceOfIValley(state, maxHeight):
    if maxHeight < 4:
        return 0

    iValleyMatrix = np.array([[1,1,1,1],
                              [0,0,0,0],
                              [1,1,1,1]])

    edgeMatrix = np.array([ [1,1,1,1],
                            [0,0,0,0]])

    h = 20 - maxHeight

    # Check edges
    leftEdge = state[np.ix_([8,9], [h, h+1, h+2, h+3])]
    if np.array_equal(leftEdge, edgeMatrix):
        return 1

    rightEdge = state[np.ix_([1,0], [h, h+1, h+2, h+3])]
    if np.array_equal(rightEdge, edgeMatrix):
        return 1

    # Check non-edges
    for i in range(8):
        subMatrix = state[np.ix_([i,i+1,i+2], [h, h+1, h+2, h+3])]
        if np.array_equal(subMatrix, iValleyMatrix):
            return 1

    return 0

if __name__ == '__main__':
    # TESTING
    testState = np.zeros((10, 20))
    testState[9][19] = 1
    testState[8][19] = 1
    testState[7][19] = 1
    testState[6][19] = 1
    testState[5][19] = 0
    testState[4][19] = 1
    testState[3][19] = 0
    testState[2][19] = 1
    testState[1][19] = 1
    testState[0][19] = 0

    testState[9][18] = 0
    testState[8][18] = 1
    testState[7][18] = 0
    testState[6][18] = 1
    testState[5][18] = 0
    testState[4][18] = 1
    testState[3][18] = 1
    testState[2][18] = 1
    testState[1][18] = 1
    testState[0][18] = 0

    testState[8][17] = 1
    testState[6][17] = 1
    testState[5][17] = 0
    testState[4][17] = 1
    testState[1][17] = 1

    testState[8][16] = 1
    testState[6][16] = 1
    testState[5][16] = 1
    testState[4][16] = 1
    testState[1][16] = 1

    print(testState)

    numCompletedLines = numCompletedLines(testState)
    numHoles = numHoles(testState)
    maxHeight, avgHeight = heights(testState)
    bumpiness = bumpiness(testState)
    iValley = existanceOfIValley(testState, maxHeight)

    print("Completed lines:", numCompletedLines)
    print("Holes:", numHoles)
    print("Max height:", maxHeight)
    print("Average height", avgHeight)
    print("Bumpiness:", bumpiness)
    print("Existance of I Valley:", iValley)
