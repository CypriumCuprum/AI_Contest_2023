import numpy
import pygame
import os
import abc
import numpy as np
import random
import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import time as t
from collections import Counter
from copy import deepcopy

#ROOT = os.path.abspath(os.path.dirname(__file__))

# PARAMETERS

GRID_WIDTH = 10
GRID_DEPTH = 20

BLOCK_LENGTH = 4
BLOCK_WIDTH = 4

FPS = 100

SCREENWIDTH  = 800
SCREENHEIGHT = 600

SPEED_UP = 10

MAX_TIME = 1000000

PIECE_SHAPE_NUM = 4
COLLIDE_DOWN_COUNT = 80.0 / SPEED_UP

ROTATE_FREQ = 10.0 / SPEED_UP

FALL_DOWN_FREQ = 40.0 / SPEED_UP

NATRUAL_FALL_FREQ = 80.0 / SPEED_UP

MOVE_SHIFT_FREQ = 10.0 / SPEED_UP

MOVE_DOWN_FREQ = 10.0 / SPEED_UP

COMBO_COUNT_FREQ = 30.0 / SPEED_UP

TSPIN_FREQ = 80.0 / SPEED_UP

Tetris_FREQ = 80.0 / SPEED_UP

BACK2BACK_FREQ = 80.0 / SPEED_UP

MAX_COMBO = 10

ipieces = [[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
          [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
opieces = [[[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]]]
jpieces = [[[0, 3, 3, 0], [0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 3, 3, 3], [0, 3, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 3, 3], [0, 0, 0, 0]],
          [[0, 0, 0, 3], [0, 3, 3, 3], [0, 0, 0, 0], [0, 0, 0, 0]]]
lpieces = [[[0, 0, 4, 0], [0, 0, 4, 0], [0, 4, 4, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 4, 4, 4], [0, 0, 0, 4], [0, 0, 0, 0]],
          [[0, 0, 4, 4], [0, 0, 4, 0], [0, 0, 4, 0], [0, 0, 0, 0]],
          [[0, 4, 0, 0], [0, 4, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0]]]
zpieces = [[[0, 5, 0, 0], [0, 5, 5, 0], [0, 0, 5, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 5, 5, 0], [5, 5, 0, 0], [0, 0, 0, 0]],
          [[0, 5, 0, 0], [0, 5, 5, 0], [0, 0, 5, 0], [0, 0, 0, 0]],
          [[0, 0, 5, 5], [0, 5, 5, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
spieces = [[[0, 0, 6, 0], [0, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 6, 6, 0], [0, 0, 6, 6], [0, 0, 0, 0]],
          [[0, 0, 6, 0], [0, 6, 6, 0], [0, 6, 0, 0], [0, 0, 0, 0]],
          [[6, 6, 0, 0], [0, 6, 6, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
tpieces = [[[0, 0, 7, 0], [0, 7, 7, 0], [0, 0, 7, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 7, 7, 7], [0, 0, 7, 0], [0, 0, 0, 0]],
          [[0, 0, 7, 0], [0, 0, 7, 7], [0, 0, 7, 0], [0, 0, 0, 0]],
          [[0, 0, 7, 0], [0, 7, 7, 7], [0, 0, 0, 0], [0, 0, 0, 0]]]
lspieces = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8] #this is the lines sent piece aka garbage lines

PIECES_DICT = {
    'I': ipieces, 'O': opieces, 'J': jpieces,
    'L': lpieces, 'Z': zpieces, 'S': spieces,
    'T': tpieces, 'G': lspieces
}

PIECE_NUM2TYPE = {1: 'I', 2: 'O', 3: 'J', 4: 'L', 5: 'Z', 6: 'S', 7: 'T', 8: 'G'}
PIECE_TYPE2NUM = {val: key for key, val in PIECE_NUM2TYPE.items()}
POSSIBLE_KEYS = ['I', 'O', 'J', 'L', 'Z', 'S', 'T']


def put_block_in_grid(grid, block, px, py):
    feasibles = block.return_pos_color(px, py)

    for x, y, c in feasibles:
        '''
        TODO: y boundary
        '''
        if -1 < x < GRID_WIDTH and -1 < y < len(grid[0]):
            grid[x][y] = c


def collide(grid, block, px, py):
    feasibles = block.get_feasible()

    # print(px)
    # print(block)
    # excess = len(grid[0]) - GRID_DEPTH
    for pos in feasibles:
        # print(px + pos[0], py + pos[1])
        if px + pos[0] > GRID_WIDTH - 1:  # right
            return True

        if px + pos[0] < 0:  # left
            return True

        if py + pos[1] > len(grid[0]) - 1:  # down
            return True

        if py + pos[1] < 0:  # up
            continue

        if grid[px + pos[0]][py + pos[1]] > 0:
            # print(px, py)
            # print(px + pos[0], py + pos[1])
            # print("Touch")
            return True

    return False


# collidedown function
# for i in range 4(y position)
# if px+y=20 then collidedown =true
# used for move down and rotation collisions
def collideDown(grid, block, px, py):
    return collide(grid, block, px, py + 1)


# collideleft function
# for i in range 4(x positions)
# if blockx +x =0 then collide left = True
# used for moving block and rotation collision
def collideLeft(grid, block, px, py):
    return collide(grid, block, px - 1, py)


# collideright function
# for i in range 4(x positions)
# if blockx +x +1>9 then collide left = True
# plus 1 is there cuz pxis on left of the piece
# used for moving block and rotation collision
def collideRight(grid, block, px, py):
    return collide(grid, block, px + 1, py)


# rotatecollision function
# when respective rotate buttons are pressed
# this function checks if collide(left right or down has occured)
# if it hasnt then rotation occurs
def rotateCollide(grid, block, px, py):
    feasibles = block.get_feasible()

    left_most = 100
    right_most = 0
    up_most = 100
    down_most = 0

    for pos in feasibles:
        right_most = max(right_most, pos[0])
        left_most = min(left_most, pos[0])

        down_most = max(down_most, pos[1])
        up_most = min(up_most, pos[1])

    c = Counter()
    # print(px)
    # print(block)
    excess = len(grid[0]) - GRID_DEPTH
    for pos in feasibles:
        # print(px + pos[0], py + pos[1])
        if px + pos[0] > 9:  # right
            c.update({"right": 1})

        if px + pos[0] < 0:  # left
            c.update({"left": 1})

        if py + pos[1] > len(grid[0]) - 1:  # down
            c.update({"down": 1})

        # if py + pos[1] < excess:   # up
        #     c.update({"up": 1})

        if 0 <= px + pos[0] <= 9 and excess <= py + pos[1] <= len(grid[0]) - 1:

            if grid[px + pos[0]][py + pos[1]] > 0:
                if pos[0] == left_most:
                    c.update({"left": 1})
                elif pos[0] == right_most:
                    c.update({"right": 1})
                elif pos[1] == down_most:
                    c.update({"down": 1})
                # elif pos[1] == up_most:
                #     c.update({"up": 1})

    # print(c)
    if len(c) == 0:
        return False
    else:
        return c.most_common()[0][0]


# this function checks if a tspin has occured
# checks all possible tspin positions
# then spins the t piece into the spot
def tspinCheck(grid, block, px, py):
    if collideDown(grid, block, px, py) == True:
        if block.block_type() == 'T':
            if px + 2 < GRID_WIDTH and py + 3 < len(grid[0]):
                if grid[px][py + 1] > 0 and grid[px][py + 3] > 0 and grid[px + 2][py + 3] > 0:

                    return True
                elif grid[px][py + 3] > 0 and grid[px + 2][py + 3] > 0 and grid[px + 2][py + 1] > 0:

                    return True
    return False


# this function rotates the piece
# when rotation button is hit the next grid in the piece list becomes the piece
def rotate(grid, block, px, py, _dir=1):
    # print(grid)

    block.rotate(_dir)

    # b = block.now_block()

    collision = rotateCollide(grid, block, px, py)  # checks for collisions
    # print(collision)
    find = 0

    if collision == "left":
        y_list = [0, 1, -1]
        for s_x in range(0, 3):
            for s_y in y_list:
                if not find and not collide(grid, block, px + s_x, py + s_y):
                    px += s_x
                    py += s_y
                    find = 1
    elif collision == "right":
        y_list = [0, 1, -1]
        for s_x in reversed(range(-2, 0)):
            for s_y in y_list:
                if not find and not collide(grid, block, px + s_x, py + s_y):
                    px += s_x
                    py += s_y
                    find = 1
    elif collision == "down":
        # y_list = [-1, -2]
        x_list = [0, -1, 1, -2, 2]
        for s_y in reversed(range(-1, 0)):
            for s_x in x_list:
                if not find and not collide(grid, block, px + s_x, py + s_y):
                    px += s_x
                    py += s_y
                    find = 1

    elif collision == "up":
        x_list = [0, -1, 1, -2, 2]
        for s_y in range(1, 2):
            for s_x in x_list:
                if not find and not collide(grid, block, px + s_x, py + s_y):
                    px += s_x
                    py += s_y
                    find = 1

    if collision != False and not find:
        block.rotate(- _dir)

        # print(collision)

    tspin = 0
    if tspinCheck(grid, block, px, py) == True:
        tspin = 1
       # print("Tspin rotate")

    # return [block, px, py, tspin]

    return block, px, py, tspin


# this function drops the piece as far as it can go until
# it collides with a piece below it
def hardDrop(grid, block, px, py):
    y = 0
    x = 0
    if collideDown(grid, block, px, py) == False:
        x = 1
    if x == 1:
        while True:
            py += 1
            y += 1
            if collideDown(grid, block, px, py) == True:
                break

    return y


# this function enables you to hold a piece
def hold(block, held, _buffer):
    # when piece is held the block at pos[0]
    # in the nextlist becomes the newpiece
    if held == None:
        held = block
        block = _buffer.new_block()

    # the piece switches with the held piece
    else:
        block, held = held, block

    return [block, held]


def freeze(last_time):
    start = t.time()
    while t.time() - start < last_time:
        pass


def get_infos(board):
    # board is equal to grid

    # borrow from https://github.com/scuriosity/machine-learning-Tetris/blob/master/Tetris.py
    # This function will calculate different parameters of the current board

    # Initialize some stuff
    heights = [0] * len(board)
    diffs = [0] * (len(board) - 1)
    holes = 0
    diff_sum = 0

    # Calculate the maximum height of each column
    for i in range(0, len(board)):  # Select a column
        for j in range(0, len(board[0])):  # Search down starting from the top of the board
            if int(board[i][j]) > 0:  # Is the cell occupied?
                heights[i] = len(board[0]) - j  # Store the height value
                break

    # Calculate the difference in heights
    for i in range(0, len(diffs)):
        diffs[i] = heights[i + 1] - heights[i]

    # Calculate the maximum height
    max_height = max(heights)

    # Count the number of holes
    for i in range(0, len(board)):
        occupied = 0  # Set the 'Occupied' flag to 0 for each new column
        for j in range(0, len(board[0])):  # Scan from top to bottom
            if int(board[i][j]) > 0:
                occupied = 1  # If a block is found, set the 'Occupied' flag to 1
            if int(board[i][j]) == 0 and occupied == 1:
                holes += 1  # If a hole is found, add one to the count

    height_sum = sum(heights)

    for i in diffs:
        diff_sum += abs(i)

    return height_sum, diff_sum, max_height, holes


class Piece(object):
    def __init__(self, _type, possible_shapes):

        self._type = _type
        self.possible_shapes = possible_shapes

        self.current_shape_id = 0

    def block_type(self):
        return self._type

    def reset(self):
        self.current_shape_id = 0

    def return_pos_color(self, px, py):
        feasibles = []

        block = self.now_block()

        for x in range(BLOCK_WIDTH):
            for y in range(BLOCK_LENGTH):
                if block[x][y] > 0:
                    feasibles.append([px + x, py + y, block[x][y]])
        return feasibles

    def return_pos(self, px, py):
        feasibles = []

        block = self.now_block()

        for x in range(BLOCK_WIDTH):
            for y in range(BLOCK_LENGTH):
                if block[x][y] > 0:
                    feasibles.append([px + x, py + y])
        return feasibles

    def get_feasible(self):
        feasibles = []

        b = self.now_block()

        for x in range(BLOCK_WIDTH):
            for y in range(BLOCK_LENGTH):
                if b[x][y] > 0:
                    feasibles.append([x, y])

        return feasibles

    def now_block(self):
        return self.possible_shapes[self.current_shape_id]

    # def move_right(self, unit=1):
    #     self.px += unit

    # def move_left(self, unit=1):
    #     self.px -= unit

    # def move_up(self, unit=1):
    #     self.py -= unit

    # def move_down(self, unit=1):
    #     self.py += unit

    def rotate(self, _dir=1):
        self.current_shape_id += _dir
        self.current_shape_id %= len(self.possible_shapes)


class Buffer(object):
    '''
    Stores the coming pieces, every 7 pieces in a group.
    '''

    def __init__(self):
        self.now_list = []
        self.next_list = []

        self.fill(self.now_list)
        self.fill(self.next_list)

    '''
    make sure "now list" are filled

                     now list           next list
    next piece <- [           ]   <-  [            ]

    '''

    def new_block(self):
        out = self.now_list.pop(0)
        self.now_list.append(self.next_list.pop(0))

        if len(self.next_list) == 0:
            self.fill(self.next_list)

        return out

    def fill(self, _list):
        pieces_keys = deepcopy(POSSIBLE_KEYS)
        random.shuffle(pieces_keys)

        for key in pieces_keys:
            _list.append(Piece(key, PIECES_DICT[key]))


'''

class for player

'''


class Player(object):
    def __init__(self, info_dict):
        self._id = info_dict.get("id")

        self._drop = info_dict.get("drop")
        self._hold = info_dict.get("hold")
        self._rotate_right = info_dict.get("rotate_right")
        self._rotate_left = info_dict.get("rotate_left")
        self._down = info_dict.get("down")
        self._left = info_dict.get("left")
        self._right = info_dict.get("right")

    @property
    def id(self):
        return self._id

    @property
    def drop(self):
        return self._drop

    @property
    def hold(self):
        return self._hold

    @property
    def rotate_right(self):
        return self._rotate_right

    @property
    def rotate_left(self):
        return self._rotate_left

    @property
    def down(self):
        return self._down

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right


'''

class Judge

'''

"""
class Judge(object):

    @staticmethod
    def check_ko_win(Tetris, max_ko):
        if Tetris.KO >= max_ko:
            return 1

        return 0

    @staticmethod
    def who_win(Tetris_1, Tetris_2):
        if Tetris_2.KO > Tetris_1.KO:  # Checks who is the winner of the game
            return Tetris_2.get_id()  # a is screebn.copy,endgame ends the game,2 is player 2 wins
        if Tetris_1.KO > Tetris_2.KO:
            return Tetris_1.get_id()  # a is screebn.copy,endgame ends the game,1 is player 1 wins
        if Tetris_1.KO == Tetris_2.KO:
            if Tetris_2.sent > Tetris_1.sent:
                return Tetris_2.get_id()  # a is screebn.copy,endgame ends the game,2 is player 2 wins
            elif Tetris_1.sent > Tetris_2.sent:
                return Tetris_1.get_id()  # a is screebn.copy,endgame ends the game,1 is player 1 wins
            elif Tetris_1.get_maximum_height() > Tetris_2.get_maximum_height():
                return Tetris_2.get_id()
            elif Tetris_2.get_maximum_height() > Tetris_1.get_maximum_height():
                return Tetris_1.get_id()
            else:
                return Tetris_1.get_id()  # no UI of draw
"""
class Tetris(object):
    def __init__(self, player, gridchoice):

        if gridchoice == "none":
            self.o_grid = [[0] * GRID_DEPTH for i in range(GRID_WIDTH)]
        """
        if gridchoice == "classic":
            self.o_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        if gridchoice == "comboking":
            self.o_grid = [[0, 0, 0, 0, 0, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [0, 0, 0, 0, 0, 6, 6, 6, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 4, 5],
                           [0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                           [0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
                           [0, 0, 0, 0, 0, 6, 6, 6, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 4, 5],
                           [0, 0, 0, 0, 0, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]

        if gridchoice == "lunchbox":
            self.o_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 5, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 2, 2, 2, 2, 2, 5, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 4, 4, 4, 4, 2, 5, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 2, 4, 4, 4, 4, 2, 5, 6],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 2, 2, 2, 2, 2, 2, 5, 6],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 5, 5, 5, 5, 5, 5, 5, 6],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]
        """
        self.player = player

        self.reset()

    def reset(self):
        self.grid = deepcopy(self.o_grid)

        self.oldko = 0  # these two used to keep track of ko's

        self._n_used_block = 1

        self.buffer = Buffer()
        # list of the held piece
        self.held = None
        self.block = self.buffer.new_block()

        # amount of lines sent for p1 and p2
        self.sent = 0
        self.tempsend = 0  # tempsending for p1 and p2
        self.oldcombo = self.combo = -1  # used for checking comboas
        self.tspin = 0  # for t spin
        self.now_back2back = 0
        self.pre_back2back = 0
        self.Tetris = 0

        # for "KO"
        self._KO = 0

        self._attacked = 0
        self._is_fallen = 0

        self.px = 4
        self.py = -2

        # DEFINING VARIABLES
        self.cleared = 0
        self.kocounter = 0
        self.stopcounter = 0

        self.isholded = 0

        self.pressedRight = False
        self.pressedLeft = False
        self.pressedDown = False

        self.LAST_ROTATE_TIME = 0
        self.LAST_MOVE_SHIFT_TIME = 0
        self.LAST_MOVE_DOWN_TIME = 0
        self.LAST_COMBO_DRAW_TIME = 0
        self.LAST_Tetris_DRAW_TIME = 0
        self.LAST_TSPIN_DRAW_TIME = 0
        self.LAST_BACK2BACK_DRAW_TIME = 0
        self.LAST_NATRUAL_FALL_TIME = 0
        self.LAST_FALL_DOWN_TIME = 0

        self.Tetris_drawing = 0
        self.tspin_drawing = 0
        self.back2back_drawing = 0

        self.combo_counter = 0

        self.natural_down_counter = 0

    def increment_timer(self):
        self.LAST_ROTATE_TIME += 1
        self.LAST_MOVE_SHIFT_TIME += 1
        self.LAST_MOVE_DOWN_TIME += 1
        self.LAST_COMBO_DRAW_TIME += 1
        self.LAST_Tetris_DRAW_TIME += 1
        self.LAST_TSPIN_DRAW_TIME += 1
        self.LAST_BACK2BACK_DRAW_TIME += 1
        self.LAST_NATRUAL_FALL_TIME += 1
        self.LAST_FALL_DOWN_TIME += 1

    @property
    def is_fallen(self):
        return self._is_fallen

    @property
    def n_used_block(self):
        return self._n_used_block

    @property
    def KO(self):
        return self._KO

    @property
    def attacked(self):
        return self._attacked

    def get_grid(self):
        excess = len(self.grid[0]) - GRID_DEPTH
        return_grids = np.zeros(shape=(GRID_WIDTH, GRID_DEPTH), dtype=np.float32)

        block, px, py = self.block, self.px, self.py
        excess = len(self.grid[0]) - GRID_DEPTH
        b = block.now_block()

        for i in range(len(self.grid)):
            return_grids[i] = np.array(self.grid[i][excess:GRID_DEPTH] + [1 for i in range(excess)], dtype=np.float32)
        return_grids[return_grids > 0] = 1

        add_y = hardDrop(self.grid, self.block, self.px, self.py)

        for x in range(BLOCK_WIDTH):
            for y in range(BLOCK_LENGTH):
                if b[x][y] > 0:
                    # draw ghost grid
                    if -1 < px + x < 10 and -1 < py + y + add_y - excess < 20:
                        return_grids[px + x][py + y + add_y - excess] = 0.3

                    if -1 < px + x < 10 and -1 < py + y - excess < 20:
                        return_grids[px + x][py + y - excess] = 0.7

        informations = np.zeros(shape=(len(PIECE_NUM2TYPE) - 1, GRID_DEPTH), dtype=np.float32)
        if self.held != None:
            informations[PIECE_TYPE2NUM[self.held.block_type()] - 1][0] = 1

        nextpieces = self.buffer.now_list
        for i in range(5):  # 5 different pieces
            _type = nextpieces[i].block_type()
            informations[PIECE_TYPE2NUM[_type] - 1][i + 1] = 1
        informations[PIECE_TYPE2NUM[self.block.block_type()] - 1][6] = 1
        # index start from 6

        informations[0][7] = self.sent / 100
        informations[1][7] = self.combo / 10
        informations[2][7] = self.pre_back2back
        informations[3][7] = self._attacked / GRID_DEPTH
        # informations[3][8] = self.time / MAX_TIME

        return_grids = np.concatenate((return_grids, informations), axis=0)

        return np.transpose(return_grids, (1, 0))

    def get_board(self):
        excess = len(self.grid[0]) - GRID_DEPTH
        return_grids = np.zeros(shape=(GRID_WIDTH, GRID_DEPTH), dtype=np.float32)

        block, px, py = self.block, self.px, self.py
        excess = len(self.grid[0]) - GRID_DEPTH
        # b = block.now_block()

        for i in range(len(self.grid)):
            return_grids[i] = np.array(self.grid[i][excess:GRID_DEPTH], dtype=np.float32)
        return_grids[return_grids > 0] = 1
        # for x in range(BLOCK_WIDTH):
        #     for y in range(BLOCK_LENGTH):
        #         if b[x][y] > 0:
        #             if -1 < px + x < 10 and -1 < py + y - excess < 20:
        #                 return_grids[px + x][py + y - excess] = 0.5

        return return_grids

    def get_maximum_height(self):
        max_height = 0
        for i in range(0, len(self.grid)):  # Select a column
            for j in range(0, len(self.grid[0])):  # Search down starting from the top of the board
                if int(self.grid[i][j]) > 0:  # Is the cell occupied?
                    max_height = max(max_height, len(self.grid[0]) - j)
                    break
        return max_height

    def reset_pos(self):
        self.px = 4
        self.py = -2 + len(self.grid[0]) - GRID_DEPTH

    def get_id(self):
        return self.player.id

    def add_attacked(self, attacked):
        self._attacked += attacked
        self._attacked = min(self._attacked, GRID_DEPTH)

    def natural_down(self):
        if self.LAST_NATRUAL_FALL_TIME >= NATRUAL_FALL_FREQ:
            if collideDown(self.grid, self.block, self.px, self.py) == False:
                self.stopcounter = 0
                # self.block.move_down()
                self.py += 1
                # pass

            self.LAST_NATRUAL_FALL_TIME = 0
        # else:
        #     self.natural_down_counter += 1

    def trigger(self, evt):
        # if (hasattr(evt, "key")):
        #     print(evt.key)
        if evt.type == pygame.KEYDOWN:
            if evt.key == self.player.rotate_right and self.LAST_ROTATE_TIME >= ROTATE_FREQ:  # rotating
                self.block, self.px, self.py, self.tspin = rotate(self.grid, self.block, self.px, self.py, _dir=1)
                self.LAST_ROTATE_TIME = 0

            if evt.key == self.player.rotate_left and self.LAST_ROTATE_TIME >= ROTATE_FREQ:  # rotating
                self.block, self.px, self.py, self.tspin = rotate(self.grid, self.block, self.px, self.py, _dir=-1)
                self.LAST_ROTATE_TIME = 0

            if evt.key == self.player.drop:  # harddrop
                y = hardDrop(self.grid, self.block, self.px, self.py)  # parameters
                # self.block.move_down(y)
                self.py += y
                # self.stopcounter = COLLIDE_DOWN_COUNT
                # self.LAST_FALL_DOWN_TIME = -FALL_DOWN_FREQ
                self.LAST_FALL_DOWN_TIME = FALL_DOWN_FREQ

            if evt.key == self.player.hold:  # holding

                if not self.isholded:
                    self.block, self.held = hold(self.block, self.held, self.buffer)  # parameters
                    self.held.reset()
                    self.reset_pos()
                    self.isholded = 1

            if evt.key == self.player.right:
                self.pressedRight = True

            if evt.key == self.player.left:
                self.pressedLeft = True

            if evt.key == self.player.down:
                self.pressedDown = True

        if evt.type == pygame.KEYUP:

            if evt.key == self.player.right:
                self.pressedRight = False

            if evt.key == self.player.left:
                self.pressedLeft = False

            if evt.key == self.player.down:
                self.pressedDown = False

    # move function
    # when respective buttons are pressed
    def move(self):
        # if keys[self.right]:
        if self.pressedRight and self.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
            if collideRight(self.grid, self.block, self.px, self.py) == False:
                self.LAST_MOVE_SHIFT_TIME = 0

                # self.block.move_right()
                self.px += 1

        if self.pressedLeft and self.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
            if collideLeft(self.grid, self.block, self.px, self.py) == False:
                self.LAST_MOVE_SHIFT_TIME = 0

                # self.block.move_left()
                self.px -= 1

        if self.pressedDown and self.LAST_MOVE_DOWN_TIME > MOVE_DOWN_FREQ:
            if collideDown(self.grid, self.block, self.px, self.py) == False:
                self.LAST_MOVE_DOWN_TIME = 0
                # self.stopcounter = 0

                # self.block.move_down()
                self.py += 1

    def check_fallen(self):
        if collideDown(self.grid, self.block, self.px, self.py) == True:
            # self.stopcounter += 1
            # if self.LAST_FALL_DOWN_TIME >= FALL_DOWN_FREQ:
            self._is_fallen = 1
            put_block_in_grid(self.grid, self.block, self.px, self.py)
            # print("fallen")

            return True

        else:
            self._is_fallen = 0
            # self.stopcounter = 0
            self.LAST_FALL_DOWN_TIME = 0

        return False

        # if self.stopcounter >= COLLIDE_DOWN_COUNT: # adds adequate delay
        #     if block_in_grid(self.grid, self.block):
        #         self.is_fallen = 1
        #         return True

        # return False

    # compute the scores when the block is fallen down.
    # return True if the computation is done.

    def compute_scores(self, cleared, combo, tspin, Tetris, pre_back2back):

        if cleared == 0:
            scores = 0
        else:
            scores = cleared if cleared == 4 else cleared - 1

            # scores from combos
            if combo > 0:
                if combo <= 8:
                    combo_scores = int((combo + 1) / 2)
                else:
                    combo_scores = 4
            else:
                combo_scores = 0

            scores += combo_scores

            # 2 line tspin
            if tspin and cleared == 2:
                scores += 3

            if pre_back2back:
                if tspin or Tetris:
                    scores += 2

        return scores

    def clear(self):

        cleared = 0

        # self.Tetris = 0

        is_combo = 0

        for y in reversed(range(GRID_DEPTH)):
            y = -(y + 1)
            row = 0  # starts checking from row zero
            for x in range(GRID_WIDTH):
                if self.grid[x][y] > 0 and self.grid[x][y] < 8:
                    row += 1

            if row == GRID_WIDTH:
                cleared += 1
                for i in range(GRID_WIDTH):
                    del self.grid[i][y]  # deletes cleared lines
                    self.grid[i] = [0] + self.grid[i]  # adds a row of zeros to the grid

        if cleared >= 1:  # for sending lines
            self.combo += 1
            if cleared == 4:  # a Tetris
                self.Tetris = 1
            else:
                self.Tetris = 0

            self.pre_back2back = self.now_back2back
        else:
            self.combo = -1
            self.Tetris = 0

        # compute scores
        scores = self.compute_scores(cleared, self.combo, self.tspin, self.Tetris, self.pre_back2back)

        if cleared >= 1:
            if self.tspin or self.Tetris:
            #    print("next backtoback")
                self.now_back2back = 1
            else:
                self.now_back2back = 0
        # print(self.pre_back2back, self.now_back2back)
        # self.tspin = 0

        self.cleared = cleared
        self.sent += scores

        real_attacked = max(0, self._attacked - scores)

        self.build_garbage(self.grid, real_attacked)

        self._attacked = 0

        return scores

        # return scores

    def check_KO(self):
        is_ko = False
        # if your grid hits the top ko = true
        excess = len(self.grid[0]) - GRID_DEPTH

        for i in range(GRID_WIDTH):
            if self.grid[i][excess] > 0:
                is_ko = True
                break

        return is_ko

    def clear_garbage(self):
        garbage = 0
        # excess = len(grid[0]) - GRID_DEPTH
        for y in range(0, len(self.grid[0])):
            for x in range(GRID_WIDTH):
                if self.grid[x][y] == 8:
                    garbage += 1
                    self.grid[x].pop(y)
                    self.grid[x] = [0] + self.grid[x]

    def build_garbage(self, grid, attacked):
        garbage_size = min(attacked, GRID_DEPTH)
        for y in range(0, garbage_size):
            for i in range(GRID_WIDTH):
                # del player.grid[i][y] # deletes top of grid
                grid[i] = grid[i] + [8]  # adds garbage lines at the bottom

        # return grid

    def check_combo(self):
        return self.combo - self.oldcombo >= 1

    def new_block(self):
        self.block = self.buffer.new_block()
        self.reset_pos()
        self.isholded = 0
        self.tspin = 0
        self._n_used_block += 1

    def update_ko(self):
        self.oldko = self._KO
        self._KO += 1

    def update_combo(self):
        self.oldcombo = self.combo
        self.combo += 1


POS_LIST = [
    {
        'combo': (44, 437),
        'Tetris': (314, 477),
        'tspin': (304, 477),
        'back2back': (314, 437),
        'board': (112, 138),
        'drawscreen': (112, 138),
        'big_ko': (44, 235),
        'ko': (140, 233),
        'transparent': (110, 135),
        'gamescreen': (0, 0),
        'attack_clean': (298, 140, 3, 360),
        'attack_alarm': (298, 481, 3, 18)
    },
    {
        'combo': (415, 437),
        'Tetris': (685, 477),
        'tspin': (675, 477),
        'back2back': (685, 437),
        'board': (495, 138),
        'drawscreen': (495, 138),
        'big_ko': (426, 235),
        'ko': (527, 233),
        'transparent': (494, 135),
        'gamescreen': (0, 0),
        'attack_clean': (680, 140, 3, 360),
        'attack_alarm': (680, 481, 3, 18)
    }
]


class ComEvent:
    '''
    IO for the AI-agent, which is simulated the pygame.event
    '''

    def __init__(self):
        self._pre_evt_list = []
        self._now_evt_list = []

    def get(self):
        return self._now_evt_list

    def set(self, actions):
        # action: list of int

        self._now_evt_list = []

        for evt in self._pre_evt_list:
            if evt.type == pygame.KEYDOWN or evt.type == "HOLD":
                if evt.key not in actions:
                    # if evt.key != action:
                    self._now_evt_list.append(ComEvt(pygame.KEYUP, evt.key))

        for action in actions:
            hold = 0
            for evt in self._pre_evt_list:
                if evt.key == action:
                    if evt.type == pygame.KEYDOWN or evt.type == "HOLD":
                        hold = 1
                        self._now_evt_list.append(ComEvt("HOLD", action))
            if not hold:
                self._now_evt_list.append(ComEvt(pygame.KEYDOWN, action))

        self._pre_evt_list = self._now_evt_list

    def reset(self):
        del self._pre_evt_list[:]
        del self._now_evt_list[:]


class ComEvt:
    '''
    class that score the key informations, it is used in ComEvent
    '''

    def __init__(self, type_, key_):
        self._type = type_
        self._key = key_

    @property
    def key(self):
        return self._key

    @property
    def type(self):
        return self._type


class TetrisInterface(abc.ABC):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'obs_type': ['image', 'grid']}

    #######################################
    # observation type:
    # "image" => screen shot of the game
    # "grid"  => the row data array of the game

    def __init__(self, gridchoice="none", obs_type="image", mode="rgb_array"):

        if mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        #self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))  # SCREEN is 800*600

        # images = load_imgs()

        # #self.renderer = Renderer(self.screen, images)

        self._obs_type = obs_type

        self._mode = mode

        self.time = MAX_TIME

        self._action_meaning = {
            0: "NOOP",
            1: "hold",
            2: "drop",
            3: "rotate_right",
            4: "rotate_left",
            5: "right",
            6: "left",
            7: "down"
        }

        self._n_actions = len(self._action_meaning)

        # print(self.action_space.n)

        self._action_set = list(range(self._n_actions))

        self.repeat = 1  # emulate the latency of human action

        self.myClock = pygame.time.Clock()  # this will be used to set the FPS(frames/s)

        self.timer2p = pygame.time.Clock()  # this will be used for counting down time in our game

        self.Tetris_list = []
        self.num_players = -1
        self.now_player = -1

        # whether to fix the speed cross device. Do this by
        # fix the FPS to FPS (100)
        self._fix_speed_cross_device = True
        self._fix_fps = FPS

    @property
    def action_meaning(self):
        return self._action_meaning

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def action_set(self):
        return self._action_set

    def screen_size(self):
        # return (x, y)
        return [SCREENHEIGHT, SCREENWIDTH]

    def get_screen_shot(self):
        ob = pygame.surfarray.array3d(pygame.display.get_surface())
        ob = np.transpose(ob, (1, 0, 2))
        return ob

    def get_seen_grid(self, mode="single"):
        now_player = self.now_player
        opp_player = 1 - self.now_player
        if mode == "double":
            now_player = 1 - self.now_player
            opp_player = self.now_player
        grid_1 = self.Tetris_list[now_player]["Tetris"].get_grid()
        grid_1[-1][-1] = self.time / MAX_TIME
        # print(grid_1)
        grid_2 = self.Tetris_list[opp_player]["Tetris"].get_grid()
        grid_2[-1][-1] = self.time / MAX_TIME
        grid = np.concatenate([grid_1, grid_2], axis=1)

        return grid.reshape(grid.shape[0], grid.shape[1], 1)
        # return self.Tetris_list[self.now_player]["Tetris"].get_grid().reshape(GRID_DEPTH, GRID_WIDTH, 1)

    def get_obs(self, mode="single"):
        if self._obs_type == "grid":
            return self.get_seen_grid(mode=mode)
        elif self._obs_type == "image":
            img = self.get_screen_shot()
        return img

    def random_action(self):
        return random.randint(0, self._n_actions - 1)

    def getCurrentPlayerID(self):
        return self.now_player

    def take_turns(self):
        self.now_player += 1
        self.now_player %= self.num_players
        return self.now_player

    def reward_func(self, infos):
        # define the reward function based on the given infos
        raise NotImplementedError

    def update_time(self, _time):
        # update the time clock and return the running state

        if self._fix_speed_cross_device:
            time_per_while = 1 / self._fix_fps * 1000  # transform to milisecond
        else:
            time_per_while = self.timer2p.tick()  # milisecond

        if _time >= 0:
            _time -= time_per_while * SPEED_UP
        else:
            _time = 0

        return _time

    def task_before_action(self, player):
        # set up the clock and curr_repeat_time
        # set the action to last_action if curr_repeat_time != 0

        self.timer2p.tick()  # start calculate the game time
        player["curr_repeat_time"] += 1
        player["curr_repeat_time"] %= self.repeat

    def get_true_action(self, player, action):
        if player["curr_repeat_time"] != 0:
            action = player["last_action"]

        player["last_action"] = action

        return action

    def reset(self, avatar1_path=None, avatar2_path=None, name1=None, name2=None, fontsize=40):
        # Reset the state of the environment to an initial state

        self.time = MAX_TIME
        self.now_player = random.randint(0, self.num_players - 1)
        self.total_reward = 0
        self.curr_repeat_time = 0  # denote the current repeat times
        self.last_infos = {'height_sum': 0,
                           'diff_sum': 0,
                           'max_height': 0,
                           'holes': 0,
                           'n_used_block': 0}

        for i, player in enumerate(self.Tetris_list):
            if i + 1 > self.num_players:
                break
            Tetris = player["Tetris"]
            com_event = player["com_event"]
            pos = player["pos"]
            player["curr_repeat_time"] = 0
            player["last_action"] = 0
            Tetris.reset()

            com_event.reset()
        #            #self.renderer.drawByName("gamescreen", pos["gamescreen"][0], pos["gamescreen"][1]) # blitting the main background

        #           #self.renderer.drawGameScreen(Tetris)

        #      #self.renderer.drawAvatar(img_path1=avatar1_path, img_path2=avatar2_path, name1=name1, name2=name2, fontsize=fontsize)
        #     #self.renderer.drawTime2p(self.time)

        # time goes until it hits zero
        # when it hits zero return endgame screen

        #pygame.display.flip()
        self.myClock.tick(FPS)

        ob = self.get_obs()

        return ob


class TetrisSingleInterface(TetrisInterface):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'obs_type': ['image', 'grid']}

    #######################################
    # observation type:
    # "image" => screen shot of the game
    # "grid"  => the row data array of the game

    def __init__(self, gridchoice="none", obs_type="image", mode="rgb_array"):
        super(TetrisSingleInterface, self).__init__(gridchoice, obs_type, mode)
        self.num_players = 1

        # The second player is dummy, it is used for
        # #self.renderer.drawByName("transparent", *opponent["pos"]["transparent"]) at around line 339
        for i in range(self.num_players + 1):
            info_dict = {"id": i}

            # adding the action information
            for k, v in self._action_meaning.items():
                info_dict[v] = k

            self.Tetris_list.append({
                'info_dict': info_dict,
                'Tetris': Tetris(Player(info_dict), gridchoice),
                'com_event': ComEvent(),
                'pos': POS_LIST[i],
                'curr_repeat_time': 0,
                'last_action': 0
            })

        self.reset()

    def reward_func(self, infos):

        if infos['is_fallen']:
            basic_reward = infos['scores']
            # additional_reward = 0.01 if infos['holes'] == 0 else 0

            # additional_reward = -0.51 * infos['height_sum'] + 0.76 * infos['cleared'] - 0.36 * infos['holes'] - 0.18 * infos['diff_sum']
            additional_reward = 0.76 * infos['cleared'] - 0.36 * infos['holes'] - 0.18 * infos['diff_sum']
            # additional_reward = infos['cleared'] # + (0.2 if infos['holes'] == 0 else 0)
            # return basic_reward + 0.01 * additional_reward - infos['penalty']
            return basic_reward + 1 * additional_reward + infos['reward_notdie']

        return 0

    def act(self, action):
        # Execute one time step within the environment

        end = 0
        scores = 0

        player, opponent = self.Tetris_list[self.now_player], self.Tetris_list[::-1][self.now_player]
        Tetris = player["Tetris"]
        com_event = player["com_event"]
        pos = player["pos"]

        self.task_before_action(player)

        action = self.get_true_action(player, action)

        Tetris.natural_down()

        com_event.set([action])

        for evt in com_event.get():
            Tetris.trigger(evt)

        Tetris.move()


        if Tetris.check_fallen():


            if Tetris.check_KO():

                Tetris.clear_garbage()

                # scores -= 5
                penalty_die = self.total_reward * 0.8

                end = 1

            Tetris.new_block()

        Tetris.increment_timer()

        ob = self.get_obs(mode="single")

        return ob, end


class TetrisEnv(gym.Env, abc.ABC):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array'],
                'obs_type': ['image', 'grid']}

    def __init__(self, interface, gridchoice="none", obs_type="grid", mode="rgb_array"):
        super(TetrisEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects

        # Example when using discrete actions:

        self.game_interface = interface(gridchoice=gridchoice,
                                        obs_type=obs_type,
                                        mode=mode)

        self._n_actions = self.game_interface.n_actions

        self.action_space = spaces.Discrete(self._n_actions)

        # print(self.action_space.n)

        self._action_set = self.game_interface.action_set

        self.action_meaning = self.game_interface.action_meaning

        self.seed()

        if obs_type == "image":
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=self.game_interface.screen_size() + [3], dtype=np.uint8)
        elif obs_type == "grid":
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=list(self.game_interface.get_seen_grid().shape), dtype=np.float32)

        self.reset()

    def random_action(self):
        return self.game_interface.random_action()

    def get_action_meanings(self):
        return [self.action_meaning[i] for i in self._action_set]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)

    def take_turns(self):
        return self.game_interface.take_turns()

    def reset(self, avatar1_path=None, avatar2_path=None, name1=None, name2=None, fontsize=40):

        self.accum_rewards = 0
        self.infos = {}
        # Reset the state of the environment to an initial state

        ob = self.game_interface.reset(avatar1_path=avatar1_path, avatar2_path=avatar2_path, name1=name1, name2=name2,
                                       fontsize=fontsize)
        ob, _ = self.game_interface.act(0)
        return ob


class TetrisSingleEnv(TetrisEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array'],
                'obs_type': ['image', 'grid']}

    def __init__(self, gridchoice="none", obs_type="grid", mode="rgb_array"):
        super(TetrisSingleEnv, self).__init__(TetrisSingleInterface, gridchoice, obs_type, mode)

    def step(self, action):
        # Execute one time step within the environment

        ob, end = self.game_interface.act(action)
        ob, end = self.game_interface.act(0)
        #ob, reward_noop, end, infos = self.game_interface.act(0)

        return ob, end


def initialize(obss):
    # initialize
    board = []
    holding = 0
    pieces = []
    for i in range(20):
        row = []
        for j in range(0, 10):
            row.append(obss[i][j][0])
        board.append(row[:])

    for row in range(20):
        for i in range(10):
            if board[row][i] == 0.7 or board[row][i] == 0.3:
                board[row][i] = int(0)
            else:
                board[row][i] = int(board[row][i])

    new_board = []
    for col in range(10):
        new_row = []
        for row in range(20):
            new_row.append(board[row][col])
        new_board.append(new_row[:])

    # get the holding piece
    for i in range(10, 17):
        if obss[0][i][0] == 1:
            holding = i - 9

    # get next 5 pieces
    for j in range(1, 6):
        for i in range(10, 17):
            if obss[j][i][0] == 1:
                pieces.append(i - 9)
                break
    return new_board, holding, pieces


def get_a_possible_move_list(right=0, left=0, rot_right=0, rot_left=0):
    a_possible_move_list = []
    for _ in range(rot_left):
        a_possible_move_list.append(4)
    for _ in range(rot_right):
        a_possible_move_list.append(3)
    for _ in range(right):
        a_possible_move_list.append(5)
    for _ in range(left):
        a_possible_move_list.append(6)
    a_possible_move_list.append(2)
    return a_possible_move_list


def get_rating_from_move(board, nowblock, list_move, chromosome):
    env_copy = TetrisSingleEnv()
    state2 = env_copy.reset()
    env_copy.game_interface.Tetris_list[0]["Tetris"].grid = deepcopy(board)
    key = PIECE_NUM2TYPE[nowblock]
    env_copy.game_interface.Tetris_list[0]["Tetris"].block = deepcopy(Piece(key, PIECES_DICT[key]))
    state2, done = env_copy.step(0)
    done = False
    for move in list_move:
        state2, done = env_copy.step(move)
        if done:
            break
    b2, a2, c2 = initialize(state2)
    tetris_copy = env_copy.game_interface.Tetris_list[0]["Tetris"]
    height_sum, diff_sum, max_height, holes = get_infos(b2)
    # print(tetris_copy.get_board())
    info = dict()
    info["cleared"] = tetris_copy.cleared
    info["height_sum"] = height_sum
    info["max_height"] = max_height
    info["holes"] = holes
    info["diff_sum"] = diff_sum
    info["n_used_block"] = tetris_copy.n_used_block
    # print(info)
    rating = 0
    rating += chromosome['cleared'] * info['cleared']*5
    rating += chromosome['height_sum'] * info['height_sum']
    rating += chromosome['max_height'] * info['max_height']
    rating += chromosome['holes'] * info['holes']
    rating += chromosome['diff_sum'] * info['diff_sum']
    rating += chromosome['n_used_block'] * info['n_used_block']

    if done:
        rating -= 500
    return rating


def get_best_move(board, nowblock, chromosome):
    max_left = 4
    max_right = 3
    possible_move_lists = []
    """extra"""
    if nowblock == 1:
        max_left = 4
        max_right = 2
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right))
        """rotate right: 1"""
        max_left = 6
        max_right = 3
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left, rot_right=1))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right, rot_right=1))
    elif nowblock == 2:
        max_left = 5
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right))
    elif nowblock == 3 or nowblock == 4 or nowblock == 7:
        """no rotate"""
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right))
        """rotate left = 2"""
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left, rot_left=2))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right, rot_left=2))
        """ rotate left 1"""
        max_left = 4
        max_right = 4
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left, rot_left=1))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right, rot_left=1))
        """rotate right"""
        max_left = 5
        max_right = 3
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left, rot_right=1))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right, rot_right=1))
    elif nowblock == 5 or nowblock == 6:
        """no rotate"""
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right))
        """rotate_right"""
        max_left = 5
        max_right = 3
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left, rot_right=1))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right, rot_right=1))
        """rotate left"""
        max_left = 4
        max_right = 4
        for left in range(1, max_left+1):
            possible_move_lists.append(get_a_possible_move_list(left=left, rot_left=1))
        for right in range(1, max_right+1):
            possible_move_lists.append(get_a_possible_move_list(right=right, rot_left=1))

    """extra"""
    """
    for i in range(1, max_left):
        possible_move_lists.append(get_a_possible_move_list(left=i))
        possible_move_lists.append(get_a_possible_move_list(left=i, rot_left=1))
        possible_move_lists.append(get_a_possible_move_list(left=i, rot_left=2))
        possible_move_lists.append(get_a_possible_move_list(left=i, rot_right=1))

    for i in range(1, max_right):
        possible_move_lists.append(get_a_possible_move_list(right=i))
        possible_move_lists.append(get_a_possible_move_list(right=i, rot_left=1))
        possible_move_lists.append(get_a_possible_move_list(right=i, rot_left=2))
        possible_move_lists.append(get_a_possible_move_list(right=i, rot_right=1))
    """
    best_list = []
    best = -50000000000

    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        cur_rating = get_rating_from_move(board, nowblock, cur_list, chromosome)
        if cur_rating > best:
            best = cur_rating
            best_list = cur_list

    return best_list


class Agent:
    def __init__(self, turn):
        self.list_move = []
        self.chromosome = {'height_sum': -1.3102745737267214, 'diff_sum': -0.6242034265122209,
                           'max_height': -0.01353979396793048, 'holes': -1.8486670208765474,
                           'cleared': 1.822646282925221, 'n_used_block': 1.2444244557243431,
                           'eval': 4224}

        self.first = 1
        self.piece_now = 0

    def choose_action(self, obs):
        if self.first == 1:
            self.first = self.first - 1
            board, holding, pieces = initialize(obs)
            self.piece_now = pieces[0]
            return 1

        if len(self.list_move) == 1:
            board, holding, pieces = initialize(obs)
            self.piece_now = pieces[0]

        if len(self.list_move) == 0:
            board, holding, pieces = initialize(obs)
            self.list_move = get_best_move(board=board, nowblock=self.piece_now, chromosome=self.chromosome)
        action = self.list_move.pop(0)
        return action





"""
for nowblock in range(1, 8):
    state = env_copy.reset()
    board, a, b = initialize(state)
    env_copy.game_interface.Tetris_list[0]["Tetris"].grid = deepcopy(board)
    key = PIECE_NUM2TYPE[nowblock]
    env_copy.game_interface.Tetris_list[0]["Tetris"].block = deepcopy(Piece(key, PIECES_DICT[key]))
    state, done = env_copy.step(0)
    state, done = env_copy.step(4)

    bo = []
    for i in range(20):
        row = []
        for j in range(0, 10):
            row.append(state[i][j][0])
        bo.append(row[:])
    print(bo,"\n")
"""

"""
start_time = t.time()
env11 = TetrisSingleEnv()
state = env11.reset()
now_block = 1
chromosome = {'height_sum': -1.9044895401339994, 'diff_sum': -0.3262308567522607,
              'max_height': -1.1880410493748155, 'holes': -1.8486670208765474,
              'cleared': 1.8800460868949906, 'n_used_block': -0.993451801560232, 'eval': 255}

board, holding, pieces = initialize(state)
best_move_list = get_best_move(board=board, nowblock=now_block, chromosome=chromosome)


print(t.time()-start_time)
"""

""" AGENT CLEAR X5"""