import numpy as np
from copy import deepcopy
import random
import time as t

START_IN_ROW = 4
START_IN_COL = 2

DEPTH_BOARD = 20
WIDTH_BOARD = 10

DEFAULT_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

I_PIECES = [
    [[1, 1, 1, 1]],
    [[1],
     [1],
     [1],
     [1]]
]

O_PIECES = [
    [[1, 1],
     [1, 1]],
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]]
]

L_PIECES = [
    [[0, 1],
     [0, 1],
     [1, 1]],
    [[1, 1, 1],
     [0, 0, 1]]
]

J_PIECES = [
    [[1, 1],
     [0, 1],
     [0, 1]],
    [[1, 1, 1],
     [1, 0, 0]]
]

Z_PIECES = [
    [[1, 0],
     [1, 1],
     [0, 1]]
]

S_PIECES = [
    [[0, 1],
     [1, 1],
     [1, 0]]
]

T_PIECES = [
    [[0, 1],
     [1, 1],
     [0, 1]],
    [[0, 1, 0],
     [1, 1, 1]]
]

PIECES_COLLECTION = I_PIECES + S_PIECES + O_PIECES + Z_PIECES + L_PIECES + J_PIECES


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
        if x + px > WIDTH_BOARD - 1:  # bound right
            return True

        if x + px < 0:  # bound left
            return True

        if y + py > DEPTH_BOARD - 1:  # bound bottom
            return True

        if y + py < 0:
            continue

        if board[x + px][y + py] > 0:
            return True
    return False


def depth_drop(board, piece, px, py):
    depth = 0
    while True:
        if check_collision(board, piece, px, py + depth):
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
    def __init__(self, grid):
        self.board = [[0 for j in range(DEPTH_BOARD)] for i in range(WIDTH_BOARD)]
        self.grid = DEFAULT_GRID
        self.grid = grid
        self.current_block = PIECES_COLLECTION[random.randint(0, len(PIECES_COLLECTION) - 1)]

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
        self.get_infos_from_grid(self.grid)

        # cleared
        self.cleared = 0

        # end game
        self.done = False

    def new_block(self):
        self.current_block = PIECES_COLLECTION[random.randint(0, len(PIECES_COLLECTION) - 1)]
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
            new_board[self.px + x][self.py + y] = 1
        self.board = new_board
        self.clear()
        self.new_block()
        return self.board, self.done

    def move(self, action):
        # 5: right  ~ +1
        # 6: left   ~ -1
        # 2: drop
        # print(np.transpose(np.array(self.current_block)))
        if action == 2:
            return self.drop()
        elif action == 5:
            if not check_collision(self.board, self.current_block, px=(self.px + 1), py=self.py):
                self.px += 1
        elif action == 6:
            if not check_collision(self.board, self.current_block, px=(self.px - 1), py=self.py):
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
        for row in range(row_start, row_end + 1):
            block.append(only_current_block[row][col_start:(col_end + 1)])
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
                    self.board[i] = [0] + self.board[i]
        self.cleared = clear


def get_para_from_state(board):
    heights = []
    holes = []
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 1:
                heights.append(len(board)-col)
                n_hol_in_col = 0
                for cell in board[0][col+1:]:
                    if cell == 1:
                        n_hol_in_col += 1
                holes.append(n_hol_in_col)
                break

    # height sum
    height_sum = sum(heights)

    # diff sum
    diff_sum = 0
    for i in range(1, len(heights)):
        diff_sum += abs(heights[i] - heights[i-1])

    # height max
    max_height = max(heights)

    # holes sum
    hole_sum = sum(holes)
    return height_sum, diff_sum, max_height, hole_sum


def initialize(obss):
    # initialize
    grid = []
    board = []
    for i in range(20):
        row = []
        for j in range(0, 10):
            row.append(obss[i][j][0])
        board.append(row[:])

    for i in range(10):
        row = []
        for j in range(20):
            if board[j][i] == 0.7:
                cell = 0
            elif board[j][i] == 0.3:
                cell = 3
            else:
                cell = int(board[j][i])
            row.append(cell)
        grid.append(row[:])
    return grid


def get_a_possible_move_list(right=0, left=0):
    a_possible_move_list = []
    for _ in range(right):
        a_possible_move_list.append(5)
    for _ in range(left):
        a_possible_move_list.append(6)
    a_possible_move_list.append(2)
    return a_possible_move_list


def get_rating_from_move(grid, list_move, chromosome):
    tetris = Tetris(grid)
    done = False
    state_board = tetris.board
    for one_move in list_move:
        state_board, done = tetris.move(one_move)
    height_sum, diff_sum, max_height, holes = get_para_from_state(state_board)
    info = dict()
    info["cleared"] = tetris.cleared
    info["height_sum"] = height_sum
    info["max_height"] = max_height
    info["holes"] = holes
    info["diff_sum"] = diff_sum
    # print(info)
    rating = 0
    rating += chromosome['cleared'] * info['cleared']
    rating += chromosome['height_sum'] * info['height_sum']
    rating += chromosome['max_height'] * info['max_height']
    rating += chromosome['holes'] * info['holes']
    rating += chromosome['diff_sum'] * info['diff_sum']
    # rating += chromosome['n_used_block'] * info['n_used_block']

    if done:
        rating -= 500
    return rating


def get_best_move(board, chromosome, rotate):
    max_left = 6
    max_right = 6
    possible_move_lists = []

    for i in range(1, max_left):
        possible_move_lists.append(get_a_possible_move_list(left=i))

    for i in range(1, max_right):
        possible_move_lists.append(get_a_possible_move_list(right=i))

    best_list = []
    best = -50000000000
    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        cur_rating = get_rating_from_move(board, cur_list, chromosome)
        if cur_rating > best:
            best = cur_rating
            best_list = cur_list
    for _ in range(rotate):
        best_list.insert(0, 4)
    return best_list, best


# rotate left: 4


class Agent:
    def __init__(self, turn):
        self.list_move = []
        self.chromosome = {
            'height_sum': -1.3102745737267214, 'diff_sum': -0.6242034265122209,
            'max_height': -0.01353979396793048, 'holes': -1.8486670208765474,
            'cleared': 1.822646282925221, 'n_used_block': 1.2444244557243431,
            'eval': 4224}
        self.rotate_left = 0
        self.best_score = -500000
        self.start = 0

    def choose_action(self, obs):
        if self.start == 0:
            self.start += 1
            return 0
        if self.rotate_left < 4:
            fixed_board = initialize(obs)
            best_list, best = get_best_move(fixed_board, self.chromosome, self.rotate_left)
            if best > self.best_score:
                self.list_move = deepcopy(best_list)
                self.best_score = best
            self.rotate_left += 1
            return 4
        action = 0
        if len(self.list_move) > 0:
            action = self.list_move.pop(0)
        if action == 2:
            self.rotate_left = 0
            self.best_score = -50000
        return action

"""
obs = [
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.3, 0.3, 0.3, 0, 0, 0]
]

agent = Agent(0)
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
print(agent.choose_action(obs))
"""