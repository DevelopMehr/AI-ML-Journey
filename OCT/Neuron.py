import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))       # squash any number into 0..1

x = np.array([1,4])    # input one customer's features
w = np.array([2, -0.5])    # weights: how much each input matters
b = 1                  # bias: a starting small push

z = np.dot(x,w) + b     # the score

print("score", z)
print("probability", sigmoid(z))