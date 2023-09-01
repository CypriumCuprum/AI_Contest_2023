import random
from copy import deepcopy
from tetris_env import TetrisDoubleEnv, TetrisSingleEnv
import json
from tetris import get_infos
import numpy as np
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


def get_a_possible_move_list(right=0, left=0, rot_right=0, rot_left=0):
    a_possible_move_list = []
    for _ in range(right):
        a_possible_move_list.append(5)
    for _ in range(left):
        a_possible_move_list.append(6)
    for _ in range(rot_left):
        a_possible_move_list.append(4)
    for _ in range(rot_right):
        a_possible_move_list.append(3)
    a_possible_move_list.append(2)
    return a_possible_move_list


def get_rating_from_move(env, list_move, chromosome):
    env_copy = TetrisSingleEnv()
    env_copy.game_interface.tetris_list[0]["tetris"] = deepcopy(env.game_interface.tetris_list[0]["tetris"])
    # height_sum, diff_sum, max_height, holes = get_infos(tetris.get_board())
    done = False
    for move in list_move:
        state, reward, done, _ = env_copy.step(move)
        if done:
            break
    tetris_copy = env_copy.game_interface.tetris_list[0]["tetris"]
    height_sum, diff_sum, max_height, holes = get_infos(tetris_copy.get_board())
    info = dict()
    info["cleared"] = tetris_copy.cleared
    info["height_sum"] = height_sum
    info["max_height"] = max_height
    info["holes"] = holes
    info["diff_sum"] = diff_sum
    info["n_used_block"] = tetris_copy.n_used_block
    """
    info = dict()
    info["cleared"] = tetris_copy.cleared
    info["height_sum"] = env_copy.game_interface.last_infos["height_sum"]
    info["max_height"] = env_copy.game_interface.last_infos["max_height"]
    info["holes"] = env_copy.game_interface.last_infos["holes"]
    info["diff_sum"] = env_copy.game_interface.last_infos["diff_sum"]
    info["n_used_block"] = env_copy.game_interface.last_infos["n_used_block"]
    """
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


def get_best_move(env, chromosome):
    max_left = 5
    max_right = 5
    possible_move_lists = []

    for i in range(1, max_left):
        possible_move_lists.append(get_a_possible_move_list(left=i))
        possible_move_lists.append(get_a_possible_move_list(left=i, rot_left=1))
        possible_move_lists.append(get_a_possible_move_list(left=i, rot_right=1))

    for i in range(1, max_right):
        possible_move_lists.append(get_a_possible_move_list(right=i))
        possible_move_lists.append(get_a_possible_move_list(right=i, rot_left=1))
        possible_move_lists.append(get_a_possible_move_list(right=i, rot_right=1))

    best_list = []
    best = -50000000000
    for cur_list in possible_move_lists:
        # env_copy = env.copy()
        # env will be copied in get_rating_from_move() function, so env local will be not changed
        cur_rating = get_rating_from_move(env, cur_list, chromosome)
        if cur_rating > best:
            best = cur_rating
            best_list = cur_list
    return best_list


def eval_chromosome(chromosome):
    env = TetrisSingleEnv()
    state = env.reset()
    done = False
    num_move_max = 500
    num_move = 1
    score = 0
    while not done and num_move < num_move_max:
        best_move_list = get_best_move(env, chromosome)
        for action in best_move_list:
            state, reward, done, _ = env.step(action)
        clear = env.game_interface.tetris_list[0]['tetris'].cleared
        print(np.transpose(env.game_interface.tetris_list[0]['tetris'].get_board()))
        if clear >= 2:
            score += clear * 5
        elif clear == 1:
            score += 1
        num_move += 1
        print(done)

    chromosome['eval'] = score


full_gen = []
population = [random_chromosome() for _ in range(num_chromosome)]
full_gen.append(population)
filename = 'rs.json'

with open(filename, 'w') as obj_file:
    for generation_ in range(10):
        print("Gen: ", generation_)
        num = 1
        for chromosome_ in population:
            eval_chromosome(chromosome_)
            print('\t', num, chromosome_)
            num += 1
        json.dump(population, obj_file, indent=2)
        population = create_new_population(population)
        full_gen.append(population)


