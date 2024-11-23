import numpy as np


class Agent:
    def __init__(self, c, d, dim, fitness_function):
        self.__c = c
        self.__d = d
        self.__dim = dim
        self.__fitness_function = fitness_function
        self.coordinates = np.empty(dim + 1)

    def fill_random(self):
        coordinates = np.random.uniform(self.__c, self.__d, self.__dim)
        fitness_value = self.__fitness_function(coordinates)
        self.coordinates = np.append(coordinates, fitness_value)
        return self.coordinates

    def update_position(self, x_random_walk, e_random_walk, min_value, max_value, dim):
        self.coordinates[dim] = np.clip((x_random_walk + e_random_walk) / 2, min_value, max_value)
