import numpy as np

def loss(y, p):
    # y = asli jawab (1 ya 0), p = model ka probability
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# (asli jawab, model ka probability)
for y, p in [(1, 0.99), (1, 0.8), (1, 0.1), (0, 0.1), (0, 0.9)]:
    print("y =", y, "| p =", p, "| loss =", round(float(loss(y, p)), 3))