#Import Pennylane

import pennylane as qml

#Import other libraries

import numpy as np
from numpy.random import randn, randint
import matplotlib.pyplot as plt

# Create a model

# Number of qubits
n_qubits = 4

# Number of layers within the quantum circuit
q_depth = 2

# Declare the device (Here we use in-built simulator, with 10000 shots)
dev = qml.device("default.qubit", wires=n_qubits, shots=10000)


def init_data():
    # Create the data

    # Number training data samples
    N = 10000

    mu = 1
    sigma = 1
    data = np.random.lognormal(mean=mu, sigma=sigma, size=N)
    np.random.shuffle(data)

    # Put data into bins

    data_pre = np.round(data)
    data = data_pre[data_pre <= 16]

    bins = np.linspace(0, 15, num=16 )
    bin_indices = np.digitize(data, bins) - 1
    data_temp = ((np.arange(16) == bin_indices[:,None]).astype(int))

    # Make it into a probability distribution

    data_dist = np.sum(data_temp, axis=0) / N

    return data_dist


@qml.qnode(dev)
def qnode(weights):

    # Init distribution
    # We start with uniform distribution
    for a in range(n_qubits):
        qml.Hadamard(wires=a)

    # Variational circuit
    # Linear Entangling layers
    for i in range(q_depth):
        for j in range(n_qubits):
            qml.RY(weights[2*(i*n_qubits + j)], wires=j)
            qml.RZ(weights[2*(i*n_qubits + j) + 1], wires=j)
        for l in range(n_qubits):
            if (l == (n_qubits - 1)):
                qml.CNOT(wires=[l,0])
            else:
                qml.CNOT(wires=[l,l+1])

    for k in range(n_qubits):
        qml.RY(weights[(2*q_depth * n_qubits) + k ], wires=k)
        qml.RZ(weights[(2*q_depth * n_qubits) + k + 1], wires=k)

    # Measurement
    # We want the probability distribution
    return qml.probs(wires=range(n_qubits))


def loss_function(model, data, weights):
    probs = model(weights)
    return np.sum(np.abs(np.log(probs) ** (-1)) - data)


if __name__ == '__main__':
    from genetic_optimizer import GeneticOptimizer
    # data = init_data()
    # weights = np.random.uniform(0, 2*math.pi, 2*(q_depth + 1)*n_qubits)
    # print(loss_function(data, weights))
    average_cost_list = []
    n_epochs = 10
    data = init_data()
    opt = GeneticOptimizer(qnode, data, loss_function)
    for i in range(1, n_epochs + 1):
        if i % 10 == 0: print("Running... Current step: ", i)
        # current_average_cost = opt.run_epoch()
        # average_cost_list.append(current_average_cost)

    plt.plot(average_cost_list)
    plt.xlabel("Epochs")
    plt.ylabel("Average cost among population")
    plt.title("Genetic algorithm")
