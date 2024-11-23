import random
from Agent import *


class AntLionOptimizer:
    def __init__(self, n, dimension, c, d, fitness_function, max_iter):
        self.__n = n
        self.__dimension = dimension
        self.__c = c
        self.__d = d
        self.__fitness_function = fitness_function
        self.__max_iter = max_iter
        self.__elite = None
        self.__agents = []

    def __initialize_population(self):
        for i in range(self.__n):
            agent = Agent(self.__c, self.__d, self.__dimension, self.__fitness_function)
            agent.fill_random()
            self.__agents.append(agent)

    def __initialize_ant_lions(self):
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
        fitness_sum = np.abs(fitness).sum()
        selection_probabilities = np.abs(fitness) / fitness_sum
        return np.random.choice(len(fitness), p=selection_probabilities)

    def __get_i_ratio(self, current_iter):
        i_ratio = 1
        if current_iter > 0.10 * self.__max_iter:
            i_ratio = 1 + (10 ** 2) * (current_iter / self.__max_iter)
        if current_iter > 0.50 * self.__max_iter:
            i_ratio = 1 + (10 ** 3) * (current_iter / self.__max_iter)
        if current_iter > 0.75 * self.__max_iter:
            i_ratio = 1 + (10 ** 4) * (current_iter / self.__max_iter)
        if current_iter > 0.90 * self.__max_iter:
            i_ratio = 1 + (10 ** 5) * (current_iter / self.__max_iter)
        if current_iter > 0.95 * self.__max_iter:
            i_ratio = 1 + (10 ** 6) * (current_iter / self.__max_iter)
        return i_ratio

    def __update_bounds(self, ant_lion, i_ratio):
        c = self.__c / i_ratio
        d = self.__d / i_ratio
        min_values = ant_lion[:-1] + c
        max_values = ant_lion[:-1] + d
        return min_values, max_values

    @staticmethod
    def __normalize_walk(walk, min_value, max_value):
        walk_min, walk_max = min(walk), max(walk)
        normalized_walk = ((walk - walk_min) * (max_value - min_value) /
                           (walk_max - walk_min) + min_value)
        return normalized_walk

    def __create_random_walks(self, i_ratio, ant_lion):
        random_steps = np.random.choice([1, -1], size=(self.__max_iter + 1, self.__dimension))
        walks = np.cumsum(random_steps, axis=0)
        min_values, max_values = self.__update_bounds(ant_lion, i_ratio)
        walks = (walks - walks.min(axis=0)) / (walks.max(axis=0) - walks.min(axis=0))
        walks = walks * (max_values - min_values) + min_values
        return walks

    @staticmethod
    def __replace_ant_lion_if_fitter(ant, ant_lion):
        if ant[-1] < ant_lion[-1]:
            return ant
        else:
            return ant_lion

    def __update_elite_if_fitter(self, ant_lions):
        best_ant_lion = ant_lions[np.argmin(ant_lions[:, -1])]
        if best_ant_lion[-1] < self.__elite[-1]:
            self.__elite = best_ant_lion
        else:
            ant_lions[np.argmin(ant_lions[:, -1])] = self.__elite

    def optimize(self):
        self.__initialize_population()
        ant_lions = self.__initialize_ant_lions()
        self.__elite = self.__find_best_ant_lion(ant_lions)

        for iteration in range(0, self.__max_iter):
            i_ratio = self.__get_i_ratio(iteration)

            for i in range(0, self.__n):
                roulette_index = self.__select_ant_lion_using_roulette(ant_lions[:, -1])
                agent = self.__agents[i]
                ant_walks = self.__create_random_walks(i_ratio, ant_lions[roulette_index])
                elite_walks = self.__create_random_walks(i_ratio, self.__elite)

                agent.update_position(ant_walks[iteration, :], elite_walks[iteration, :])

                ant_lions[roulette_index] = self.__replace_ant_lion_if_fitter(agent.coordinates,
                                                                              ant_lions[roulette_index])

            self.__update_elite_if_fitter(ant_lions)
            if iteration % 50 == 0:
                print(f"Iteration {iteration}, Elite fitness: {self.__elite[-1]}")
        return self.__elite[:-1], self.__elite[-1]
