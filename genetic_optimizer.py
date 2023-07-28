import numpy as np


class GeneticOptimizer:

    def __init__(self, G, model_class, population_size=10, num_cubits=1, coef_bits=8):
        self.G = G
        self.ModelClass = model_class
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


if __name__ == '__main__':
    go = GeneticOptimizer()