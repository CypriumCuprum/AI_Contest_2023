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


# SETTINGS 2

ipieces = [[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
          [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
opieces = [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]]
"""opieces = [[[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]],
          [[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]]"""
jpieces = [[[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
          [[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
lpieces = [[[0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
          [[0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
zpieces = [[[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
          [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
spieces = [[[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
          [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
          [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
tpieces = [[[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]],
          [[0, 0, 1, 0], [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]

NUM_PIECES = [1, 2, 3, 4, 5, 6, 7]
MAP_NUM_PIECE = {1: ipieces, 2: opieces, 3: jpieces, 4: lpieces, 5: zpieces, 6: spieces, 7: tpieces}


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
    def __init__(self):
        self.board = [[0 for j in range(DEPTH_BOARD)] for i in range(WIDTH_BOARD)]
        self.index_block = random.randint(1, 7)
        self.current_block = MAP_NUM_PIECE[self.index_block][0]
        self.sub_index_block = 0
        self.next_blocks = [1, 2, 3, 4, 5, 6, 7]
        # Position default
        self.px = 4
        self.py = 0

        # cleared
        self.cleared = 0

        # end game
        self.done = False

    def new_block(self):
        self.index_block = self.next_blocks.pop(0)
        if self.next_blocks == []:
            blocks = deepcopy(NUM_PIECES)
            random.shuffle(blocks)
            self.next_blocks = blocks[:5]
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
            if board[row][i] == np.float32(0.7) or board[row][i] == np.float32(0.3):
                board[row][i] = int(0)
            elif board[row][i] == 0:
                board[row][i] = int(0)
            else:
                board[row][i] = int(1)

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


def get_possible_move_lists(possible_move_lists, nowblock):
    max_left = 4
    max_right = 3
    """extra"""
    newpossible_movelists = []
    for item in possible_move_lists:
        new_item = deepcopy(item)
        if nowblock == 1:
            """ no rotate"""
            max_left = 4
            max_right = 2
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
            """rotate right: 1"""
            max_left = 6
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_right=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_right=1))
        elif nowblock == 2:
            max_left = 5
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
        elif nowblock == 3 or nowblock == 4 or nowblock == 7:
            """no rotate"""
            max_left = 4
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
            """rotate left = 2"""
            max_left = 4
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_left=2))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_left=2))
            """ rotate left 1"""
            max_left = 4
            max_right = 4
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_left=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_left=1))
            """rotate right"""
            max_left = 5
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_right=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_right=1))
        elif nowblock == 5 or nowblock == 6:
            """no rotate"""
            max_left = 4
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right))
            """rotate_right"""
            max_left = 5
            max_right = 3
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_right=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_right=1))
            """rotate left"""
            max_left = 4
            max_right = 4
            for left in range(0, max_left + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(left=left, rot_left=1))
            for right in range(1, max_right + 1):
                newpossible_movelists.append(new_item + get_a_possible_move_list(right=right, rot_left=1))
    return newpossible_movelists


def get_rating_from_move(board, list_block, list_move, gen):
    game_check = Tetris()
    game_check.board = board
    game_check.clear()
    extra_height = game_check.cleared
    game_check.cleared = 0
    game_check.index_block = list_block[0]
    game_check.next_blocks = list_block[1:]
    game_check.current_block = MAP_NUM_PIECE[game_check.index_block][0]
    done = False
    state_board = game_check.board
    for one_move in list_move:
        state_board, done = game_check.move(one_move)
        if done == True:
            break
    info = game_check.get_info_from_state()
    rating = 0
    if game_check.cleared == 1:
        rating -= 30
    for i in range(len(info)):
        rating += info[i]*gen[i]
    if info[2] + extra_height >= 16:
        rating -= 200
    if done:
        rating -= 200
    return rating


def get_best_move(board, list_block, gen):
    possible_move_lists = [[]]
    for block in list_block:
        new_list = get_possible_move_lists(possible_move_lists, block)
        possible_move_lists = deepcopy(new_list)
    best_list = []
    best = -20000000000
    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        cur_rating = get_rating_from_move(board, list_block, cur_list, gen)
        if cur_rating > best:
            best = cur_rating
            best_list = deepcopy(cur_list)
    return best_list


# _________________________________________________GENERATIONNN_____________________________________________
num_chromosome = 9
num_gen = 250


def get_random_cof(a, b):
    return random.uniform(a, b)


def random_gen():
    gen = []
    """
    height_sum, diff_sum, max_height, hole_sum, deepest_unfilled,
    blocks, col_holes, cleared_num, pit_hole_percent
    """
    gen.append(get_random_cof(-2, 0))  # height_sum              0
    gen.append(get_random_cof(-2, 0))  # diff_dum                1
    gen.append(get_random_cof(-2, 0))  # max_height              2
    gen.append(get_random_cof(-2, 0))  # hole_sum                3
    gen.append(get_random_cof(-2, 2))  # deepest_unfilled        4
    gen.append(get_random_cof(-2, 2))  # blocks                  2
    gen.append(get_random_cof(-2, 0))  # col_holes               6
    gen.append(get_random_cof(0, 2))  # cleared_num              7
    gen.append(get_random_cof(-2, 2))  # pit_hole_percent        8
    # score
    gen.append(0)
    # move
    gen.append(0)
    return gen


def best_score(gen):
    return gen[num_chromosome]


def cross_over(dad, mum, cross_over_rate=0.3):
    child = random_gen()
    for i in range(num_chromosome):
        if random.random() < cross_over_rate:
            child[i] = mum[i]
        else:
            child[i] = dad[i]
    child[num_chromosome] = min(mum[num_chromosome], dad[num_chromosome])
    child[num_chromosome + 1] = min(mum[num_chromosome+1], dad[num_chromosome+1])
    return child


def mutate(gen, mutate_rate=0.3):
    gen_new = deepcopy(gen)
    for i in range(num_chromosome):
        if random.random() < mutate_rate:
            if i == 4 or i == 5 or i == 8:
                gen_new[i] = get_random_cof(-2, 0)
            elif i == 7:
                gen_new[i] = get_random_cof(0, 2)
            else:
                gen_new[i] = get_random_cof(-2, 2)
    return gen_new


def selection(old_population, rate=0.4):
    sorted_old_population = list(reversed(sorted(old_population, key=best_score)))
    num_chosen = int(num_gen*rate)
    new_population = sorted_old_population[:num_chosen]
    return new_population


def create_new_population(old_population):
    new_population = selection(old_population)
    now_length = len(new_population)
    while len(new_population) < num_gen:
        # choose mom and dad
        index_1 = random.randint(0, now_length-1)
        index_2 = random.randint(0, now_length-1)
        while index_2 == index_1:
            index_2 = random.randint(0, now_length-1)

        mum = new_population[index_1]
        dad = new_population[index_2]

        # crossover
        child = cross_over(mum, dad)

        # mutation
        child_mutation = mutate(child)

        # add to new population
        new_population.append(child_mutation)

    return new_population



def eval_gene(gene):
    game_tetris = Tetris()
    max_move = 500
    done = False
    cnt_move = 0
    evaluate = 0
    lines_cleared_1 = 0
    lines_cleared_2 = 0
    lines_cleared_3 = 0
    lines_cleared_4 = 0

    while not done and cnt_move < max_move:
        cnt_move += 1
        board = deepcopy(game_tetris.board)
        list_block = [game_tetris.index_block] + [game_tetris.next_blocks[0]]
        list_move = get_best_move(board, list_block, gene)
        for one_move in list_move:
            board, done = game_tetris.move(one_move)
            if done:
                break
        if game_tetris.cleared == 1:
            lines_cleared_1 += 1
        if game_tetris.cleared == 2:
            lines_cleared_2 += 1
            evaluate += 4
        if game_tetris.cleared == 3:
            lines_cleared_3 += 1
            evaluate += 7
        if game_tetris.cleared == 4:
            lines_cleared_4 += 1
            evaluate += 11
    print(lines_cleared_1)
    print(lines_cleared_2)
    print(lines_cleared_3)
    print(lines_cleared_4)
    gene[num_chromosome] += evaluate
    gene[num_chromosome+1] += cnt_move

"""
population = [random_gen() for _ in range(num_gen)]

for gen in range(10):

    print("Generation: ", gen)
    for gene in population:
        eval_gene(gene)
        print(gene)
    population = create_new_population(population)
    max_move += 1000
    print("\n")
"""
gen = [-1.5758211071560368, -0.08359013311115282, -1.434217250807808, -0.3294210379853766, 0.028193731783810794, 1.599865072325608, -1.5967463265727169, 1.351534619961469, 1.9588249502582844, 0, 0]

gen2 = [-0.45871856486636053, -0.3211088438566083, -0.5221084651717414, -1.4317746855089484, -0.8248875964891864, 1.856227466507848, -1.59595104909416, 0.7765780163107312, 1.8598908998668202, 0, 0]
eval_gene(gen2)
print(gen2)

gen3 = [-1.871611780086036, -0.4839019224082466, -0.04302682587711093, -1.4921366328180081, 0.5522848539415994, 0.014898950750871887, -0.9354735387638364, 0.7779382470511451, 1.9589147178686788, 0, 0]


#eval_gene(gen3)
print(gen3)

# NOW BLOCK IS 3x3
