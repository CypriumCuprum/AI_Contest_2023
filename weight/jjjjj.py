import json

file_name = 'savegen1.json'
file_name_2 = 'save_gen0.json'


def compare(chromosome):
    return chromosome['eval']


with open(file_name_2, 'r') as obf:
    popu2 = json.load(obf)
    popu = list(reversed(popu2))

with open(file_name, 'r') as obj_f:
    population = json.load(obj_f)

for i in popu:
    population.append(i)
    if len(population) == 50:
        break

with open(file_name, 'w') as obj:
    json.dump(population, obj, indent= 2)


