import random
import time

import numpy as np

from tetris_new import *


num_chromosome = 9
num_gen = 500


def get_random_cof(a, b):
    return random.uniform(a, b)


def random_gen():
    gen = []
    """
    height_sum, diff_sum, max_height, hole_sum, deepest_unfilled,
    blocks, col_holes, cleared_num, pit_hole_percent
    """
    gen.append(get_random_cof(-2, 2))  # height_sum              0
    gen.append(get_random_cof(-2, 2))  # diff_dum                1
    gen.append(get_random_cof(-2, 0))  # max_height              2
    gen.append(get_random_cof(-2, 0))  # hole_sum                3
    gen.append(get_random_cof(-2, 2))  # deepest_unfilled        4
    gen.append(get_random_cof(-2, 2))  # blocks                  2
    gen.append(get_random_cof(-2, 0))  # col_holes               6
    gen.append(get_random_cof(0, 2))  # cleared_num              7
    gen.append(get_random_cof(-2, 2))  # pit_hole_percent        8
    # score
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
    return child


def mutate(gen, mutate_rate=0.3):
    gen_new = deepcopy(gen)
    for i in range(num_chromosome):
        if random.random() < mutate_rate:
            if i == 2 or i == 3 or i == 6:
                gen_new[i] = get_random_cof(-2, 0)
            elif i == 7:
                gen_new[i] = get_random_cof(0, 2)
            else:
                gen_new[i] = get_random_cof(-2, 2)
    return gen_new


def selection(old_population, rate=0.2):
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


# Get move
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


def eval_gene(gene):
    game_tetris = Tetris()
    max_move = 10000
    done = False
    cnt_move = 0
    eval = 0

    while not done and cnt_move < max_move:
        rotate = 0
        best_list = []
        best = -2000
        while rotate < 4:
            moves, score = get_best_move(game_tetris, gene, rotate)
            if score > best:
                best_list = deepcopy(moves)
                best = score
            game_tetris.move(4)
            rotate += 1
        for i in best_list:
            state, done = game_tetris.move(i)
        if game_tetris.cleared >= 2:
            eval += game_tetris.cleared
        eval += game_tetris.cleared
        cnt_move += 1
    gene[num_chromosome] += eval
    gene[num_chromosome+1] += cnt_move

"""
population = [random_gen() for _ in range(num_gen)]
for generation in range(20):
    print("Generation: ", generation)

    for gen in population:
        eval_gene(gen)
        print(gen)

    population = create_new_population(population)
    print("\n")


"""
# height_sum, diff_sum, max_height, hole_sum, deepest_unfilled,
#    blocks, col_holes, cleared_num, pit_hole_percent
"""
gen = [-0.21197420768695463, -0.24403906746598647, -0.8312194209124975, -1.6320763454016523, 0.23178672994481664, 1.790423658211798, -0.6438507478734055, 1.518327151413257, 1.7956848170127273, 0, 0]
eval_gene(gen)
print(gen)

gen2 = [0.14656848647051968, -0.47777763373223125, -0.6555745669992836, -1.6386328872651454, -0.47058381771977853, 0.2670299008025454, -1.3615585370798318, 1.492955930575336, -1.2133566834515155, 0, 0]
eval_gene(gen2)
print(gen2)
"""
"""
TEST_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    ]
tetrisgame = Tetris(TEST_GRID)
li = tetrisgame.get_info_from_state()
print(li)
"""

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

print(ipieces)
print(opieces)
print(jpieces)
print(lpieces)
print(zpieces)
print(spieces)
print(tpieces)