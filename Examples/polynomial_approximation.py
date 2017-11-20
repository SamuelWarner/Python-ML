"""
Example of neural network learning a polynomial equation. Test polynomial is f(x) = (6x^2 + 3x) ÷ (3x)

Training is run on x values from 1.0 to 100.0
"""
from mazex import MazeX
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Create list to store how close networks guesses are
graph_data = []

# Create Neural Network
net = MazeX([1, 20, 4, 1], ["relu", "relu", 'lin'], learning_constant=0.00001)

# test how close the network is to the correct answer given x = 12 and log the result for the graph
def check(run):
    guess = net.forward(np.array([[12.0]]))
    print(f"run {run} OFF BY: {25 - guess[0][0]}")
    graph_data.append(25 - guess[0][0])


# run a bunch of training steps on random values to help network learn the polynomial
for i in range(100):
    t = random.uniform(1.0, 100.0)
    ans = ((6 * math.pow(t, 2)) + (3 * t)) / (3 * t)
    Y = np.full((1, 1), ans)
    X = np.full((1, 1), t)

    net.train(X, Y)
    check(i)

# plot the training data for visual feedback of learning progress. Saves graph to same directory as script
plt.plot(graph_data)
plt.ylabel('Error')
plt.xlabel("training run")
plt.title('Error over time')
plt.savefig(f'Polynomial_approximation.png')



