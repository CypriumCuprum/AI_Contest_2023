from settingsnew import *
import numpy as np
from copy import deepcopy
import random


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
        self.index_block = random.randint(1, 7)
        self.current_block = MAP_NUM_PIECE[self.index_block][0]
        self.sub_index_block = 0
        self.next_blocks = NUM_PIECES
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
        self.sub_index_block += 1
        self.sub_index_block %= 4
        self.current_block = MAP_NUM_PIECE[self.index_block][self.sub_index_block]

    def rotate_left(self):
        self.sub_index_block += 3
        self.sub_index_block %= 4
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
        pit = (WIDTH_BOARD*DEPTH_BOARD - height_sum)
        pit_hole_percent = pit/(pit+hole_sum)

        return [height_sum, diff_sum, max_height, hole_sum, deepest_unfilled,
                blocks, col_holes, cleared_num, pit_hole_percent]

    def get_possible_move(self):
        max_left = self.px
        max_right = WIDTH_BOARD - self.px - len(self.current_block)
        full_move = []
        for i in range(0, max_left+1):
            full_move.append(get_a_possible_move_list(left=i))
        for i in range(1, max_right+1):
            full_move.append(get_a_possible_move_list(right=i))
        return full_move


