#Import Pennylane

import pennylane as qml

#Import other libraries

import numpy as np
from numpy.random import randn, randint
import os
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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

    # plt.hist(bin_indices, bins = 16)

    # Make it into a probability distribution

    data_dist = np.sum(data_temp, axis=0) / N

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # y = range(16)
    # ax.bar(y, data_dist, alpha=0.5)
    # plt.show()

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


def loss_function(data, weights):
    probs = qnode(weights)
    return np.sum(np.abs(np.log(probs) ** (-1)) - data)


if __name__ == '__main__':
    data = init_data()
    weights = np.random.uniform(0, 2*math.pi, 2*(q_depth + 1)*n_qubits)
    print(loss_function(data, weights))
