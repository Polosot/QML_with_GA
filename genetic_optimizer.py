import numpy as np
import random


class GeneticOptimizer:

    def __init__(self, model, data, loss_func, G=100, population_size=10, num_cubits=3, coef_bits=8):

        assert population_size % 2 == 0

        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.G = G
        self.population_size = population_size
        self.num_cubits = num_cubits
        self.coef_bits = coef_bits
        self.population = self.init_population()

    def get_random_bits(self, n):

        return ''.join([str(x) for x in np.random.randint(0, 2, n)])

    def init_model_qweights(self, num_cubits):

        res = []

        for _ in range(num_cubits):
            res.append(self.get_random_bits(self.coef_bits))  # theta
            res.append(self.get_random_bits(self.coef_bits))  # phi

        return res

    def init_population(self):
        res = []
        for _ in range(self.population_size):
            res.append(self.init_model_qweights(self.num_cubits))

        return res

    def run_model(self, population):

        losses = []
        for p in population:
            weights = [2 * np.pi * int(k, 2) / (2 ** self.coef_bits) for k in p]
            loss = self.loss_func(self.model, self.data, weights)
            losses.append(loss)

        return losses

    def selection(self, population, losses):

        population_with_losses = list(zip(population, losses))
        population_with_losses.sort(key=lambda x: x[1])

        # kill and clone
        k = len(population_with_losses) // 2
        population_with_losses[k:] = population_with_losses[:k]

        new_population = [x[0] for x in population_with_losses]

        random.shuffle(new_population)

        return new_population

    def cross_over(self, population):

        res = []
        for pair_id in range(self.population_size // 2):
            i = pair_id * 2

            chromosome_a = population[i]
            chromosome_b = population[i + 1]

            cross_over_i = np.random.randint(0, self.coef_bits)
            res.append(chromosome_a[:cross_over_i] + chromosome_b[cross_over_i:])
            res.append(chromosome_b[:cross_over_i] + chromosome_a[cross_over_i:])

        return res


if __name__ == '__main__':
    pass
