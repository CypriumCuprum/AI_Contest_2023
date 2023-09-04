import numpy as np
from copy import deepcopy
import random


START_IN_ROW = 4
START_IN_COL = 2

DEPTH_BOARD = 20
WIDTH_BOARD = 10

DEFAULT_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

I_PIECES = [
    [[1, 1, 1, 1]]
    ,
    [[1],
     [1],
     [1],
     [1]]
]

O_PIECES = [
    [[1, 1],
     [1, 1]]
]


L_PIECES = [
    [[0, 1],
     [0, 1],
     [1, 1]]
    ,
    [[1, 1, 1],
     [0, 0, 1]]
    ,
    [[1, 1],
     [1, 0],
     [1, 0]]
    ,
    [[1, 0, 0],
     [1, 1, 1]]
]

J_PIECES = [
    [[1, 1],
     [0, 1],
     [0, 1]]
    ,
    [[1, 1, 1],
     [1, 0, 0]]
    ,
    [[1, 0],
     [1, 0],
     [1, 1]]
    ,
    [[0, 0, 1],
     [1, 1, 1]]
]

Z_PIECES = [
    [[1, 0],
     [1, 1],
     [0, 1]]
    ,
    [[0, 1, 1],
     [1, 1, 0]]
]

S_PIECES = [
    [[0, 1],
     [1, 1],
     [1, 0]]
    ,
    [[1, 1, 0],
     [0, 1, 1]]
]

T_PIECES = [
    [[0, 1],
     [1, 1],
     [0, 1]]
    ,
    [[1, 1, 1],
     [0, 1, 0]]
    ,
    [[1, 0],
     [1, 1],
     [1, 0]]
    ,
    [[0, 1, 0],
     [1, 1, 1]]
]

NUM_PIECES = [1, 2, 3, 4, 5, 6, 7]
MAP_NUM_PIECE = {1: I_PIECES, 2: S_PIECES, 3: O_PIECES, 4: Z_PIECES, 5: L_PIECES, 6: J_PIECES, 7: T_PIECES}
PIECES_COLLECTION = I_PIECES + S_PIECES + O_PIECES + Z_PIECES + L_PIECES + J_PIECES + T_PIECES


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
    if len(full_pos) == 0:
        return True
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
        if check_collision(board, piece, px, py + depth) is True:
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


def get_a_possible_move_list(right=0, left=0):
    a_possible_move_list = []
    for _ in range(right):
        a_possible_move_list.append(5)
    for _ in range(left):
        a_possible_move_list.append(6)
    a_possible_move_list.append(2)
    return a_possible_move_list


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
        self.current_block = PIECES_COLLECTION[random.randint(0, len(PIECES_COLLECTION) - 1)]
        self.index_block = 7
        self.sub_index_block = 0
        self.next_blocks = [1, 2, 3, 4, 5, 6, 7]
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
        if len(self.next_blocks) == 0:
            blocks = deepcopy(NUM_PIECES)
            random.shuffle(blocks)
            self.next_blocks = deepcopy(blocks)
        self.index_block = self.next_blocks.pop(0)
        self.sub_index_block = 0
        self.current_block = MAP_NUM_PIECE[self.index_block][self.sub_index_block]
        self.px = 4
        self.py = 0

    def drop(self):
        depth_falling = depth_drop(board=self.board, piece=self.current_block, px=self.px, py=self.py)
        if depth_falling == -1:
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
        return new_board, self.done

    def rotate_right(self):
        if self.sub_index_block == len(MAP_NUM_PIECE[self.index_block]) - 1:
            self.sub_index_block = 0
        else:
            self.sub_index_block += 1
        self.current_block = MAP_NUM_PIECE[self.index_block][self.sub_index_block]

    def rotate_left(self):
        if self.sub_index_block == 0:
            self.sub_index_block = len(MAP_NUM_PIECE[self.index_block]) - 1
        else:
            self.sub_index_block -= 1
        self.current_block = MAP_NUM_PIECE[self.index_block][self.sub_index_block]

    def move(self, action):
        # 5: right  ~ +1
        # 6: left   ~ -1
        # 3: rotate right
        # 4: rotate left
        # 2: drop
        # print(np.transpose(np.array(self.current_block)))
        if action == 2:
            return self.drop()
        if action == 5:
            if not check_collision(self.board, self.current_block, px=(self.px + 1), py=self.py):
                self.px += 1
        if action == 6:
            if not check_collision(self.board, self.current_block, px=(self.px - 1), py=self.py):
                self.px -= 1
        if action == 3:
            self.rotate_right()
        if action == 4:
            self.rotate_left()
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
        self.current_block = deepcopy(block)
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

    def get_info_from_state(self):
        heights = []
        holes = []
        for row in range(WIDTH_BOARD):
            height_row = 0
            for col in range(DEPTH_BOARD):
                if self.board[row][col] == 1:
                    height_row = DEPTH_BOARD - col
                    n_hol_in_col = 0
                    for col_hole in range(col + 1, DEPTH_BOARD):
                        if self.board[row][col_hole] == 0:
                            n_hol_in_col += 1
                    holes.append(n_hol_in_col)
                    break
            heights.append(height_row)

        # height sum
        height_sum = sum(heights)
        # diff sum
        diff_sum = 0
        for i in range(1, len(heights)):
            diff_sum += abs(heights[i] - heights[i - 1])

        # height max
        max_height = max(heights)

        # holes sum
        hole_sum = sum(holes)

        # deepest unfilled
        deepest_unfilled = min(heights)

        # blocks count
        blocks = 0
        for row in self.board:
            blocks += np.count_nonzero(np.array(row))
        blocks /= 4

        # col holes
        col_holes = np.count_nonzero(np.array(holes))

        # cleared
        cleared_num = self.cleared

        # pit hole percent
        pit = (WIDTH_BOARD * DEPTH_BOARD - height_sum)
        pit_hole_percent = pit / (pit + hole_sum)

        return [height_sum, diff_sum, max_height, hole_sum, deepest_unfilled,
                blocks, col_holes, cleared_num, pit_hole_percent]

    def get_possible_move(self):
        max_left = self.px
        max_right = WIDTH_BOARD - self.px - len(self.current_block)
        full_move = []
        for i in range(0, max_left + 1):
            full_move.append(get_a_possible_move_list(left=i))
        for i in range(1, max_right + 1):
            full_move.append(get_a_possible_move_list(right=i))
        return full_move


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
            if board[j][i] == np.float32(0.7):
                cell = 0
            elif board[j][i] == np.float32(0.3):
                cell = 3
            else:
                cell = int(board[j][i])
            row.append(cell)
        grid.append(row[:])
    return grid


def get_rating_from_move(tetris, list_move, gen):
    new_tetris = deepcopy(tetris)
    done = False
    state_board = new_tetris.board
    for one_move in list_move:
        state_board, done = new_tetris.move(one_move)
        if done == True:
            break
    info = new_tetris.get_info_from_state()
    rating = 0
    for i in range(len(info)):
        rating += info[i]*gen[i]
    if done:
        rating -= 200
    return rating


def get_best_move(tetris, gen, rotate):
    new_tetris = deepcopy(tetris)
    possible_move_lists = new_tetris.get_possible_move()
    best_list = []
    best = -20000000000
    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        cur_rating = get_rating_from_move(new_tetris, cur_list, gen)
        if cur_rating > best:
            best = cur_rating
            best_list = deepcopy(cur_list)
    if rotate == 3:
        best_list.insert(0, 3)
    else:
        for _ in range(rotate):
            best_list.insert(0, 4)
    return best_list, best


class Agent:
    def __init__(self, turn):
        self.list_move = []
        self.gen = [-1.3102745737267214, -0.6242034265122209, -0.01353979396793048, -1.8486670208765474, 0, 0, 0, 1.822646282925221, 0, 0]
        self.rotate_left = 0
        self.best_score = -500000

    def choose_action(self, obs):
        if self.rotate_left < 4:
            fixed_board = initialize(obs)
            tetri_game = Tetris(fixed_board)
            best_list, best = get_best_move(tetri_game, self.gen, self.rotate_left)
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






