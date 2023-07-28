from main import loss_function, init_data, qnode
from genetic_optimizer import GeneticOptimizer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = init_data()
    go = GeneticOptimizer(model=qnode, data=data, loss_func=loss_function, num_cubits=12, coef_bits=36,
                          mutation_prob=0.001, population_size=100, G=300)

    average_cost_list = go.fit()

    plt.plot(average_cost_list)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss among population")
    plt.title("Genetic algorithm")
    plt.show()