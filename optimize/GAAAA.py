import random
from copy import deepcopy
from tetris_env import TetrisDoubleEnv, TetrisSingleEnv
import json
import numpy as np
from tetris import get_infos

num_weight = 6
num_chromosome = 50

""" 
    A chromosome is a list that has 9 features and 1 score

    Features:
        height_sum
        diff_sum
        max_height
        holes
        cleared
        n_used_block

"""


def get_random_cof(a, b):
    return random.uniform(a, b)


def random_chromosome():
    chromosome = {
        "height_sum": get_random_cof(-2, 0),
        "diff_sum": get_random_cof(-2, 0),
        "max_height": get_random_cof(-2, 0),
        "holes": get_random_cof(-2, 0),
        "cleared": get_random_cof(0, 2),
        "n_used_block": get_random_cof(-2, 2),
        "eval": 0
    }
    return chromosome


def best_score(chromosome):
    return chromosome['eval']


def cross_over(dad, mum, cross_over_rate=0.2):
    child1 = deepcopy(dad)
    child2 = deepcopy(mum)
    for key, value in child1.items():
        if key != "val" and random.random() < cross_over_rate:
            child1[key] = mum[key]
            child2[key] = dad[key]

    return child1, child2


def mutate(chromosome, mutate_rate=0.2):
    chromosome_new = deepcopy(chromosome)
    for key, value in chromosome_new.items():
        if key != "val" and random.random() < mutate_rate:
            chromosome_new[key] = get_random_cof(-1, 1)
    return chromosome_new


def selection(old_population):
    index_1 = random.randint(0, num_chromosome - 1)
    while True:
        index_2 = random.randint(0, num_chromosome - 1)
        if index_2 != index_1:
            break
    chosen_chromosome = old_population[index_1]
    if old_population[index_2]['eval'] > old_population[index_1]['eval']:
        chosen_chromosome = old_population[index_2]
    return chosen_chromosome


def create_new_population(old_population, chosen=26):
    new_population = []
    sorted_old_population = sorted(old_population, key=best_score)
    while len(new_population) < num_chromosome - chosen:
        # selection
        chosen_chromosome_1 = selection(sorted_old_population)
        chosen_chromosome_2 = selection(sorted_old_population)

        # crossover
        new_1, new_2 = cross_over(chosen_chromosome_1, chosen_chromosome_2)

        # mutation
        new_1_mu = mutate(new_1)
        new_2_mu = mutate(new_2)

        new_population.append(new_1_mu)
        new_population.append(new_2_mu)

    extra_index = num_chromosome - 1
    while len(new_population) < num_chromosome:
        new_population.append(sorted_old_population[extra_index])
        extra_index -= 1

    return new_population


"""
action meaning:
    0: "NOOP",
    1: "hold",
    2: "drop",
    3: "rotate_right",
    4: "rotate_left",
    5: "right",
    6: "left",
    7: "down"
"""

PIECE_NUM2TYPE = {1: 'I', 2: 'O', 3: 'J', 4: 'L', 5: 'Z', 6: 'S', 7: 'T'}
PIECE_TYPE2NUM = {val: key for key, val in PIECE_NUM2TYPE.items()}


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
    #board1 = deepcopy(board)
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
        for i in range(27, 34):
            if obss[j][i][0] == 1:
                pieces.append(i - 26)
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

from tetris import Piece
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


def get_rating_from_move(board, holding, list_move, chromosome):
    env_copy = TetrisSingleEnv()
    #stt = env_copy.reset()

    env_copy.game_interface.tetris_list[0]["tetris"].grid = deepcopy(board)
    key = PIECE_NUM2TYPE[holding]
    env_copy.game_interface.tetris_list[0]["tetris"].block = deepcopy(Piece(key, PIECES_DICT[key]))

    done = False
    for move in list_move:
        state2, reward, done, _ = env_copy.step(move)
        if done:
            break
    #print(np.transpose(b2))
    tetris_copy = env_copy.game_interface.tetris_list[0]["tetris"]
    height_sum, diff_sum, max_height, holes = get_infos(tetris_copy.get_board())
    #print(tetris_copy.get_board())
    info = dict()
    info["cleared"] = tetris_copy.cleared
    info["height_sum"] = height_sum
    info["max_height"] = max_height
    info["holes"] = holes
    info["diff_sum"] = diff_sum
    info["n_used_block"] = tetris_copy.n_used_block
    #print(info)
    rating = 0
    rating += chromosome['cleared'] * info['cleared']
    rating += chromosome['height_sum'] * info['height_sum']
    rating += chromosome['max_height'] * info['max_height']
    rating += chromosome['holes'] * info['holes']
    rating += chromosome['diff_sum'] * info['diff_sum']
    rating += chromosome['n_used_block'] * info['n_used_block']

    if done:
        rating -= 500
    return rating


def get_best_move(board, holding, chromosome):
    max_left = 6
    max_right = 6
    possible_move_lists = []

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

    best_list = []
    best = -50000000000
    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        cur_rating = get_rating_from_move(board, holding, cur_list, chromosome)
        if cur_rating > best:
            best = cur_rating
            best_list = cur_list
    return best_list


def eval_chromosome(chromosome):
    env1 = TetrisSingleEnv()
    state = env1.reset()
    done = False
    num_move_max = 500
    num_move = 1
    score = 0
    stat, re, a1, b1 = env1.step(1)
    stat, re, a1, b1 = env1.step(2)
    board, holding, pie = initialize(stat)
    while not done and num_move < num_move_max:
        best_move_list = get_best_move(board=board, holding=holding, chromosome=chromosome)
        ob, reward, done, _ = env1.step(1)
        #print("boo\n")
        #print(boo, "\n")

        for action in best_move_list:
            ob, reward, done, _ = env1.step(action)
        board, holding, pieces = initialize(ob)
        #print(np.transpose(board))
        clear = env1.game_interface.tetris_list[0]['tetris'].cleared
        if clear >= 2:
            score += clear * 2
        elif clear == 1:
            score += 1
        num_move += 1
        #print(done)

    chromosome['eval'] = score


full_gen = []

filename = 'rs.json'
filename1 = 'rs1.json'
with open(filename1, 'r') as f:
    population = json.load(f)

full_gen.append(population)
with open(filename, 'w') as obj_file:
    for generation_ in range(1, 10):
        population = create_new_population(population)
        print("Gen: ", generation_)
        num = 1
        for chromosome_ in population:
            eval_chromosome(chromosome_)
            print('\t', num, chromosome_)
            num += 1

        full_gen.append(population)
    json.dump(full_gen, obj_file)

"""

env = TetrisSingleEnv()
state = env.reset()

obs, a, b, c = env.step(2)
bb, board, holding, pieces = initialize(obs)
print(holding)
obs, a, b, c = env.step(1)
obs, a, b, c = env.step(2)

bb, board, holding, pieces = initialize(obs)
print(holding)
key = PIECE_NUM2TYPE[holding]
obs, a, b, c = env.step(1)
b1, board, holding, pieces = initialize(obs)
print(holding)
print(b1)
print(board)


env2 = TetrisSingleEnv()
state2 = env2.reset()

env2.game_interface.tetris_list[0]['tetris'].grid = board
#env2.game_interface.tetris_list[0]['tetris'].reset()
env2.game_interface.tetris_list[0]["tetris"].block = Piece(key, PIECES_DICT[key])
#st, z1, l, k = env2.step(0)
oba = env2.game_interface.get_obs()
bo = []
for i in range(20):
    row = []
    for j in range(0, 10):
        row.append(oba[i][j][0])
    bo.append(row[:])
#boa1, boa, hol, pie = initialize(st)
#print(boa1)
print(bo)
#print(boa)

"""

"""
{'height_sum': -1.823466031276068, 'diff_sum': -0.1792394644354034, 'max_height': -1.2236821504006368, 'holes': -0.9275614242597181, 'cleared': 0.3146450185204248, 'n_used_block': -1.9599898348706395, 'eval': 43}
"""