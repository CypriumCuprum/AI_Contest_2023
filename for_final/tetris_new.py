from settingsnew import *
import numpy as np
from copy import deepcopy
import random

"""
A square matrix is easy to rotate. Just transpose to gain a 90 degrees rotate 
TETROMINO SAMPLE
suitable for the transposed board
I: [[1,1,1,1]]
[[1]
 [1]
 [1]
 [1]]

T:
[[0,1,0]        
 [1,1,1]]
 
[[0,1]
 [1,1] 
 [0,1]]


L:
[[0,0,1]
 [1,1,1]]
 
J:
[[1,0,0]
 [1,1,1]]
 
O:
[[1,1]
 [1,1]]
 
Z:
[[1,1,0]
 [0,1,1]]

S:
[[0,1,1]
 [1,1,0]]

"""


def get_full_pos(piece):
    full_pos = []
    for x, row in enumerate(piece):
        for y, cell in enumerate(row):
            if cell == 1:
                full_pos.append((x, y))
    return full_pos


def check_collision(board, piece, px, py):
    # the top left of piece:
    # px is num of col in board
    # py is num of row in board
    full_pos = get_full_pos(piece)
    for pos in full_pos:
        x = pos[0]
        y = pos[1]
        if x+px > WIDTH_BOARD-1:  # bound right
            return True

        if x+px < 0:  # bound left
            return True

        if y+py > DEPTH_BOARD-1:  # bound bottom
            return True

        if y+py < 0:
            continue

        if board[x+px][y+py] > 0:
            return True
    return False


def depth_drop(board, piece, px, py):
    depth = 0
    while True:
        if check_collision(board, piece, px, py+depth):
            break
        depth += 1
    depth -= 1
    return depth


def get_pos_start_end(grid):
    start = 0
    for row in grid:
        if np.count_nonzero(row) == 0:
            start += 1
        else:
            break

    end = len(grid) - 1
    for row in range(len(grid) - 1, -1, -1):
        if np.count_nonzero(grid[row]) == 0:
            end -= 1
        else:
            break
    return start, end


"""
GRID SAMPLE

 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 |   [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], 
 |   [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3], 
 X   [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 |   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
\/----------------------------------Y-------------------------------->
"""


class Tetris:
    def __init__(self, grid=DEFAULT_GRID):
        self.board = [[0 for j in range(DEPTH_BOARD)] for i in range(WIDTH_BOARD)]
        self.grid = grid
        self.current_block = PIECES_COLLECTION[random.randint(0, len(PIECES_COLLECTION)-1)]

        # Position default
        self.px = 4
        self.py = 0

        # define action
        self.action_meaning = {
            2: "drop",
            5: "right",
            6: "left"
        }

        # Get block and its position from grid
        if self.grid != DEFAULT_GRID:
            self.get_infos_from_grid(self.grid)

        # cleared
        self.cleared = 0

        # end game
        self.done = False

    def new_block(self):
        self.current_block = PIECES_COLLECTION[random.randint(0, len(PIECES_COLLECTION)-1)]
        self.px = 4
        self.py = 0

    def drop(self):
        depth_falling = depth_drop(board=self.board, piece=self.current_block, px=self.px, py=self.py)
        if depth_falling == 0:
            self.done = True
            return self.board, self.done
        self.py += depth_falling
        new_board = deepcopy(self.board)
        full_pos_block = get_full_pos(self.current_block)
        for pos in full_pos_block:
            x = pos[0]
            y = pos[1]
            new_board[self.px+x][self.py+y] = 1
        self.board = new_board
        self.clear()
        self.new_block()
        return new_board, self.done

    def move(self, action):
        # 5: right  ~ +1
        # 6: left   ~ -1
        # 2: drop
        # print(np.transpose(np.array(self.current_block)))
        if action == 2:
            return self.drop()
        elif action == 5:
            if not check_collision(self.board, self.current_block, px=(self.px+1), py=self.py):
                self.px += 1
        elif action == 6:
            if not check_collision(self.board, self.current_block, px=(self.px-1), py=self.py):
                self.px -= 1
        return self.board, self.done

    def get_infos_from_grid(self, grid):
        only_current_block = []
        board = []
        for row in range(0, len(grid)):
            only_cur_bl_row = []
            row_board = []
            for col in range(0, len(grid[0])):
                only_cur_bl_row.append(1 if grid[row][col] == 3 else 0)
                row_board.append(grid[row][col] if grid[row][col] != 3 else 0)
            only_current_block.append(only_cur_bl_row)
            board.append(row_board)
        row_start, row_end = get_pos_start_end(only_current_block)
        grid_2 = np.transpose(np.array(only_current_block))
        col_start, col_end = get_pos_start_end(grid_2)
        block = []
        for row in range(row_start, row_end+1):
            block.append(only_current_block[row][col_start:(col_end+1)])
        self.current_block = block
        self.px = row_start
        self.board = board

    def clear(self):
        clear = 0
        for col in range(DEPTH_BOARD):
            cnt = 0
            for row in range(WIDTH_BOARD):
                if self.board[row][col] == 1:
                    cnt += 1
            if cnt == WIDTH_BOARD:
                clear += 1
                for i in range(WIDTH_BOARD):
                    del self.board[i][col]
                    self.board[i] = [0]+self.board[i]
        self.cleared = clear


"""
arr = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
if arr != DEFAULT_GRID:
    print(1)
print(arr)
print("")
row_start, row_end = get_pos_start_end(arr)
grid_2 = np.transpose(np.array(arr))
col_start, col_end = get_pos_start_end(grid_2)
block = []
for row in range(row_start, row_end+1):
    block.append(arr[row][col_start:(col_end+1)])

print(block)
"""

Game = Tetris(grid=TEST_GRID)
print(Game.current_block)
print(Game.board)
n_board = Game.move(2)
print(Game.board)

