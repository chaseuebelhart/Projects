import random, sys
'''
This is a headless tetris engine that is fast and multi-process safe
'''

BOXSIZE = 20
BOARDWIDTH = 10
BOARDHEIGHT = 20
WINDOWWIDTH = BOXSIZE * BOARDWIDTH
WINDOWHEIGHT = BOXSIZE * BOARDHEIGHT
BLANK = '.'

MOVESIDEWAYSFREQ = 0.15
MOVEDOWNFREQ = 0.1

XMARGIN = 0
TOPMARGIN = 0

TEMPLATEWIDTH = 5
TEMPLATEHEIGHT = 5

S_SHAPE_TEMPLATE = [['..OO.',
                     '.OO..',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..OO.',
                     '...O.',
                     '.....',
                     '.....']]

Z_SHAPE_TEMPLATE = [['.OO..',
                     '..OO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '.OO..',
                     '.O...',
                     '.....',
                     '.....']]

I_SHAPE_TEMPLATE = [['..O..',
                     '..O..',
                     '..O..',
                     '..O..',
                     '.....'],
                    ['OOOO.',
                     '.....',
                     '.....',
                     '.....',
                     '.....']]

O_SHAPE_TEMPLATE = [['.OO..',
                     '.OO..',
                     '.....',
                     '.....',
                     '.....']]

J_SHAPE_TEMPLATE = [['.O...',
                     '.OOO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..OO.',
                     '..O..',
                     '..O..',
                     '.....',
                     '.....'],
                    ['.OOO.',
                     '...O.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..O..',
                     '.OO..',
                     '.....',
                     '.....']]

L_SHAPE_TEMPLATE = [['...O.',
                     '.OOO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..O..',
                     '..OO.',
                     '.....',
                     '.....'],
                    ['.OOO.',
                     '.O...',
                     '.....',
                     '.....',
                     '.....'],
                    ['.OO..',
                     '..O..',
                     '..O..',
                     '.....',
                     '.....']]

T_SHAPE_TEMPLATE = [['..O..',
                     '.OOO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..OO.',
                     '..O..',
                     '.....',
                     '.....'],
                    ['.OOO.',
                     '..O..',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '.OO..',
                     '..O..',
                     '.....',
                     '.....']]

PIECES = {'S': S_SHAPE_TEMPLATE,
          'Z': Z_SHAPE_TEMPLATE,
          'J': J_SHAPE_TEMPLATE,
          'L': L_SHAPE_TEMPLATE,
          'I': I_SHAPE_TEMPLATE,
          'O': O_SHAPE_TEMPLATE,
          'T': T_SHAPE_TEMPLATE}


class GameState:
    def __init__(self):
        # DEBUG
        self.total_lines = 0

        # setup variables for the start of the game
        self.reinit()

    def reinit(self):
        self.board = self.getBlankBoard()
        self.movingDown = False # note: there is no movingUp variable
        self.movingLeft = False
        self.movingRight = False
        self.score = 0
        self.lines = 0
        self.height = 0
        self.level, self.fallFreq = self.calculateLevelAndFallFreq()

        self.fallingPiece = self.getNewPiece()
        self.nextPiece = self.getNewPiece()
        self.piecesPlayed = 0

        self.frame_step([1,0,0,0,0,0])

    def frame_step(self,input):
        self.movingLeft = False
        self.movingRight = False

        reward = 0
        terminal = False

        #none is 100000, left is 010000, up is 001000, right is 000100, space is 000010, q is 000001
        if self.fallingPiece == None:
            # No falling piece in play, so start a new piece at the top
            self.piecesPlayed += 1
            self.fallingPiece = self.nextPiece
            self.nextPiece = self.getNewPiece()

            if not self.isValidPosition():
                terminal = True

                self.reinit()
                # can't fit a new piece on the self.board, so game over
                return [self.fallingPiece, self.nextPiece, self.board], reward, terminal

        # moving the piece sideways
        if (input[1] == 1) and self.isValidPosition(adjX=-1):
            #self.fallingPiece['x'] -= 1
            self.movingLeft = True
            self.movingRight = False

        elif (input[3] == 1) and self.isValidPosition(adjX=1):
            #self.fallingPiece['x'] += 1
            self.movingRight = True
            self.movingLeft = False

        # rotating the piece (if there is room to rotate)
        elif (input[2] == 1):
            self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] + 1) % len(PIECES[self.fallingPiece['shape']])
            if not self.isValidPosition():
                self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] - 1) % len(PIECES[self.fallingPiece['shape']])

        elif (input[5] == 1): # rotate the other direction
            self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] - 1) % len(PIECES[self.fallingPiece['shape']])
            if not self.isValidPosition():
                self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] + 1) % len(PIECES[self.fallingPiece['shape']])

        # move the current piece all the way down
        elif (input[4] == 1):
            self.movingDown = False
            self.movingLeft = False
            self.movingRight = False
            for i in range(1, BOARDHEIGHT):
                if not self.isValidPosition(adjY=i):
                    break
            self.fallingPiece['y'] += i - 1

        # handle moving the piece because of user input
        if (self.movingLeft or self.movingRight):
            if self.movingLeft and self.isValidPosition(adjX=-1):
                self.fallingPiece['x'] -= 1
            elif self.movingRight and self.isValidPosition(adjX=1):
                self.fallingPiece['x'] += 1

        if self.movingDown:
            self.fallingPiece['y'] += 1

        # see if the piece has landed
        cleared = 0
        if not self.isValidPosition(adjY=1):
            # falling piece has landed, set it on the self.board
            self.addToBoard()

            cleared = self.removeCompleteLines()
            if cleared > 0:
                if cleared == 1:
                    self.score += 40 * self.level
                elif cleared == 2:
                    self.score += 100 * self.level
                elif cleared == 3:
                    self.score += 300 * self.level
                elif cleared == 4:
                    self.score += 1200 * self.level

            self.score += self.fallingPiece['y']

            self.lines += cleared
            self.total_lines += cleared

            reward = self.height - self.getHeight()
            self.height = self.getHeight()

            self.level, self.fallFreq = self.calculateLevelAndFallFreq()
            self.fallingPiece = None

        else:
            # piece did not land, just move the piece down
            self.fallingPiece['y'] += 1

        if cleared > 0:
            reward = 100 * cleared

        observation = [self.fallingPiece, self.nextPiece, self.board, \
            self.getHeight(), self.checkIfDropCreatesCave()]
        return observation, reward, terminal

    def getActionSet(self):
        return range(6)

    def getHeight(self):
        stack_height = 0
        for i in range(0, BOARDHEIGHT):
            blank_row = True
            for j in range(0, BOARDWIDTH):
                if self.board[j][i] != '.':
                    blank_row = False
                    break
            if not blank_row:
                stack_height = BOARDHEIGHT - i
                break
        return stack_height

    def getReward(self):
        stack_height = None
        num_blocks = 0
        for i in range(0, BOARDHEIGHT):
            blank_row = True
            for j in range(0, BOARDWIDTH):
                if self.board[j][i] != '.':
                    num_blocks += 1
                    blank_row = False
            if not blank_row and stack_height is None:
                stack_height = BOARDHEIGHT - i

        if stack_height is None:
            return BOARDHEIGHT
        else:
            return BOARDHEIGHT - stack_height
            return float(num_blocks) / float(stack_height * BOARDWIDTH)

    def isGameOver(self):
        return self.fallingPiece == None and not self.isValidPosition()

    def calculateLevelAndFallFreq(self):
        # Based on the self.score, return the self.level the player is on and
        # how many seconds pass until a falling piece falls one space.
        self.level = min(int(self.lines / 10) + 1, 10)
        self.fallFreq = 0.27 - (self.level * 0.02)
        return self.level, self.fallFreq

    def getNewPiece(self):
        # return a random new piece in a random rotation and color
        shape = random.choice(list(PIECES.keys()))
        newPiece = {'shape': shape,
                    'rotation': random.randint(0, len(PIECES[shape]) - 1),
                    'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                    'y': 0, # start it above the self.board (i.e. less than 0)
                    'color': 0}
        return newPiece

    def addToBoard(self):
        # fill in the self.board based on piece's location, shape, and rotation
        fallingPieceX = self.fallingPiece['x']
        fallingPieceY = self.fallingPiece['y']
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y][x] != BLANK:
                    self.board[x + fallingPieceX][y + fallingPieceY] = 0

    def checkIfDropCreatesCave(self):
        # Check whether dropping the current peice would create a hollow cavity

        if self.fallingPiece == None:
            return False


        fallingPieceX = self.fallingPiece['x']
        fallingPieceY = self.fallingPiece['y']
        distanceToBlockBelow = -1
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y][x] != BLANK:
                    if PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y+1][x] == BLANK:
                        # Piece in block is not above another piece in the currrent block
                        for boardY in range(y + fallingPieceY, 20):
                            # Compute y distance to nearest block below
                            if self.board[x + fallingPieceX][boardY] != '.' or boardY == 19:
                                # Found block beneath
                                thisDistanceToBlockBelow = boardY - (y + fallingPieceY)

                                if distanceToBlockBelow == -1:
                                    distanceToBlockBelow = thisDistanceToBlockBelow
                                elif thisDistanceToBlockBelow != distanceToBlockBelow:
                                    # Found a cave
                                    return True

                                break
        return False

    def getBlankBoard(self):
        # create and return a new blank self.board data structure
        self.board = []
        for i in range(BOARDWIDTH):
            self.board.append([BLANK] * BOARDHEIGHT)
        return self.board

    def isOnBoard(self,x, y):
        return x >= 0 and x < BOARDWIDTH and y < BOARDHEIGHT

    def isValidPosition(self,adjX=0, adjY=0):
        # Return True if the piece is within the self.board and not colliding
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                isAboveBoard = y + self.fallingPiece['y'] + adjY < 0
                if isAboveBoard or PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y][x] == BLANK:
                    continue
                if not self.isOnBoard(x + self.fallingPiece['x'] + adjX, y + self.fallingPiece['y'] + adjY):
                    return False
                if self.board[x + self.fallingPiece['x'] + adjX][y + self.fallingPiece['y'] + adjY] != BLANK:
                    return False
        return True

    def isCompleteLine(self, y):
        # Return True if the line filled with boxes with no gaps.
        for x in range(BOARDWIDTH):
            if self.board[x][y] == BLANK:
                return False
        return True

    def removeCompleteLines(self):
        # Remove any completed lines on the self.board, move everything above them down, and return the number of complete lines.
        numLinesRemoved = 0
        y = BOARDHEIGHT - 1 # start y at the bottom of the self.board
        while y >= 0:
            if self.isCompleteLine(y):
                # Remove the line and pull boxes down by one line.
                for pullDownY in range(y, 0, -1):
                    for x in range(BOARDWIDTH):
                        self.board[x][pullDownY] = self.board[x][pullDownY-1]
                # Set very top line to blank.
                for x in range(BOARDWIDTH):
                    self.board[x][0] = BLANK
                numLinesRemoved += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y -= 1 # move on to check next row up
        return numLinesRemoved
