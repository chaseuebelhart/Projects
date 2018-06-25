from statistics import stdev
# Format: "Feature name": (Measurement Scale, Possible values)
# FEATURES = {
            # "totalHeight":          ("number": tuple(range(201))),
            # "maxHoleHeight":        ("number": tuple(range(21))),
            # "numHoles":             ("number": tuple(range(191))),
            # "playableRow":          ("number": tuple(range(21))),
            # "numHoleRows":          ("number": tuple(range(21))),
            # "numHoleCols":          ("number": tuple(range(11))),
            # "numBlockades":         ("number": tuple(range(191)))
            # }

def total_height(board):
    height = 0
    for col in range(0, len(board)):
        for row in range(len(board[col])-1, -1, -1):
            if board[col][row]:
                height+=1
    return height

def max_hole_height(board):
    maxHoleHeight = 0
    for col in range(0, len(board)):
        emptySpace = False
        colHasHole = False
        colHeight = 0
        for row in range(len(board[col])-1, -1, -1):
            if not board[col][row]:
                emptySpace = True
            elif emptySpace:
                colHasHole = True
                colHeight += 1
            else:
                colHeight += 1
        if colHasHole and colHeight>maxHoleHeight:
            maxHoleHeight = colHeight
    return maxHoleHeight

def num_holes(board):
    numHoles = 0
    for col in range(0, len(board)):
        emptySpaces = 0
        for row in range(len(board[col])-1, -1, -1):
            if not board[col][row]:
                emptySpaces += 1
            elif emptySpaces:
                numHoles += emptySpaces
                emptySpaces = 0
    return numHoles

def playable_row(board):
    """
    Not sure what they meant by this so I'll ignore it for now
    """
    pass

def num_hole_rows(board):
    numHoleRows = 0
    for row in range(len(board[0])-1, -1, -1):
        rowHasHole = False
        emptySpace = False
        for col in range(0, len(board)):
            if not board[col][row]:
                emptySpace = True
            elif emptySpace:
                rowHasHole = True
        if rowHasHole:
            numHoleRows += 1
    return numHoleRows

def num_hole_cols(board):
    numHoleCols = 0
    for col in range(0, len(board)):
        colHasHole = False
        emptySpace = False
        for row in range(len(board[col])-1, -1, -1):
            if not board[col][row]:
                emptySpace = True
            elif emptySpace:
                colHasHole = True
        if colHasHole:
            numHoleCols += 1
    return numHoleCols

def num_blockades(board):
    numBlockades = 0
    for col in range(0, len(board)):
        colHasHole = False
        emptySpace = False
        colBlockades = 0
        for row in range(len(board[col])-1, -1, -1):
            if not board[col][row]:
                emptySpace = True
            elif emptySpace:
                colHasHole = True
            
            if colHasHole:
                colBlockades += 1
        numBlockades += colBlockades
    return numBlockades

def highest_col(board):
    max_hieght = -1
    highest_column = -1

    for col in range(0, len(board)):
        height = 0
        for row in range(len(board[col])-1, -1, -1):
            if board[col][row]:
                height+=1
        if height > max_hieght:
            max_hieght = height
            highest_column = col
    return highest_column

def absolute(n):
    if n<0:
        return n * -1
    else:
        return n

def bumpiness(board):
    cols_heights = []
    bumpiness = 0
    for col in range(0, len(board)):
        height = 0
        c_h = 0
        for row in range(len(board[col])-1, -1, -1):
            c_h += 1
            if board[col][row]:
                height += c_h
                c_h = 0
        cols_heights.append(height)
        if col>1:
            bumpiness += absolute(cols_heights[-1]-cols_heights[-2])
    return bumpiness
