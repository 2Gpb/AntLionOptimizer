import numpy as np
import random


class AntLionOptimizer:
    def __init__(self, n, dimension, c, d, fitness_function, max_iter):
        self.__n = n
        self.__dimension = dimension
        self.__c = c
        self.__d = d
        self.__fitness_function = fitness_function
        self.__max_iter = max_iter
        self.__elite = None

    def __initialize_population(self):
        population = np.zeros((self.__n, self.__dimension + 1))
        for i in range(self.__n):
            for j in range(self.__dimension):
                population[i, j] = random.uniform(self.__c, self.__d)
            population[i, -1] = self.__fitness_function(population[i, :-1])
        return population

    @staticmethod
    def __find_best_ant_lion(population):
        best_agent = population[population[:, -1].argsort()][0, :]
        return best_agent

    @staticmethod
    def __select_ant_lion_using_roulette(fitness):
        fitness_sum = 0
        for i in range(0, len(fitness)):
            fitness_sum += abs(fitness[i])
        selection_probabilities = abs(fitness) / fitness_sum
        selected_index = np.random.choice(len(fitness), p=selection_probabilities)
        return selected_index

    def __update_bounds(self, ant_lion, current_iter):
        i_ratio = 1
        if current_iter > 0.10 * self.__max_iter:
            w_exploration = 2
            i_ratio = (10 ** w_exploration) * (current_iter / self.__max_iter)
        elif current_iter > 0.50 * self.__max_iter:
            w_exploration = 3
            i_ratio = (10 ** w_exploration) * (current_iter / self.__max_iter)
        elif current_iter > 0.75 * self.__max_iter:
            w_exploration = 4
            i_ratio = (10 ** w_exploration) * (current_iter / self.__max_iter)
        elif current_iter > 0.90 * self.__max_iter:
            w_exploration = 5
            i_ratio = (10 ** w_exploration) * (current_iter / self.__max_iter)
        elif current_iter > 0.95 * self.__max_iter:
            w_exploration = 6
            i_ratio = (10 ** w_exploration) * (current_iter / self.__max_iter)
        c = self.__c / i_ratio
        d = self.__d / i_ratio
        min_values = ant_lion[:-1] + c
        max_values = ant_lion[:-1] + d
        return min_values, max_values

    def __create_random_walk(self):
        x_random_walk = [0] * (self.__max_iter + 1)
        for i in range(1, self.__max_iter + 1):
            rand = random.choice([0, 1])
            x_random_walk[i] = x_random_walk[i - 1] + (2 * rand - 1)
        return x_random_walk

    @staticmethod
    def __normalize_walk(walk, min_values, max_values, dim, iteration):
        walk_min, walk_max = min(walk), max(walk)
        normalized_walk = ((walk[iteration] - walk_min) * (max_values[dim] - min_values[dim]) /
                           (walk_max - walk_min) + min_values[dim])
        return normalized_walk

    @staticmethod
    def __update_position(ant,  x_random_walk, e_random_walk, min_value, max_value, dim):
        ant[dim] = np.clip((x_random_walk + e_random_walk) / 2,
                           min_value, max_value)
        return ant

    @staticmethod
    def __replace_ant_lion_if_fitter(ant, ant_lion):
        if ant[-1] < ant_lion[-1]:
            return ant
        else:
            return ant_lion

    def __update_elite_if_fitter(self, ant_lions):
        best_ant_lion = ant_lions[ant_lions[:, -1].argsort()][0, :]
        if best_ant_lion[-1] < self.__elite[-1]:
            self.__elite = best_ant_lion
        else:
            ant_lions[ant_lions[:, -1].argsort()][0, :] = self.__elite

    def optimize(self):
        ants = self.__initialize_population()
        ant_lions = self.__initialize_population()

        self.__elite = self.__find_best_ant_lion(ant_lions)

        for iteration in range(0, self.__max_iter):
            for i in range(self.__n):
                index = self.__select_ant_lion_using_roulette(ant_lions[:, -1])
                ant_lion = ant_lions[index]

                min_values, max_values = self.__update_bounds(ant_lion, iteration)
                elite_min_values, elite_max_values = self.__update_bounds(self.__elite, iteration)

                for j in range(0, self.__dimension):
                    ant_walk = self.__create_random_walk()
                    elite_walk = self.__create_random_walk()

                    ant_walk[iteration] = self.__normalize_walk(ant_walk, min_values, max_values, j, iteration)
                    elite_walk[iteration] = self.__normalize_walk(elite_walk, elite_min_values, elite_max_values,
                                                                  j, iteration)

                    ants[i] = self.__update_position(ants[i], ant_walk[iteration], elite_walk[iteration],
                                                     min_values[j], max_values[j], j)

                ants[i, -1] = self.__fitness_function(ants[i, :-1])
                ant_lions[index] = self.__replace_ant_lion_if_fitter(ants[i], ant_lion)
            self.__update_elite_if_fitter(ant_lions)
            # print(f"Iteration {iteration + 1}, Elite fitness: {self.__elite[-1]}")
        # print("Best Solution:", np.round(self.__elite[:-1], 5), "fitness:", round(self.__elite[-1], 5))
        return np.round(self.__elite[:-1], 5), round(self.__elite[-1], 5)
