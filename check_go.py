from main import loss_function, init_data, qnode
from genetic_optimizer import GeneticOptimizer


if __name__ == '__main__':
    data = init_data()
    go = GeneticOptimizer(model=qnode, data=data, loss_func=loss_function, num_cubits=12)

    # losses = go.run_model(go.population)
    # p = go.selection(go.population, losses)
    # p = go.cross_over(p)
    # p = go.mutate(p, 0.0001)

    go.fit()