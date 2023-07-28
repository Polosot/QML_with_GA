

class GeneticOptimizer:

    def __init__(self, G, model_class, population_size=10):
        self.G = G
        self.ModelClass = model_class
        self.population_size = population_size
        self.population = self.init_population()

    def init_model_qweights(self):
        pass

    def init_population(self):
        res = []
        for _ in range(self.population_size):
            res.append(self.init_model_qweights())

        return res



if __name__ == '__main__':
    pass