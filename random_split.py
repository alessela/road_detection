import random

shuffle_instances = list(range(289))
random.shuffle(shuffle_instances)
with open('random_split.csv', 'w+') as f:
    f.write(','.join(map(str, shuffle_instances[:260])) + '\n' + ','.join(map(str, shuffle_instances[260:])))