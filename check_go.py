from main import loss_function, init_data
from genetic_optimizer import GeneticOptimizer


if __name__ == '__main__':
    data = init_data()
    go = GeneticOptimizer(data=data, model_func=loss_function, num_cubits=12)

    losses = go.run_model(go.population)
    p = go.selection(go.population, losses)
    p = go.cross_over(p)

    print(p)