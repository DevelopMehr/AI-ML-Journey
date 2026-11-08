import numpy as np
np.random.seed(1)                                  # har baar same random numbers

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])                 # XOR

W1 = np.random.randn(2, 4)     # 2 inputs -> 4 hidden neurons
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)     # 4 hidden -> 1 output
b2 = np.zeros((1, 1))
lr = 1.0

for step in range(5001):
    # FORWARD: aage ki taraf
    h = sigmoid(X @ W1 + b1)                       # hidden layer
    p = sigmoid(h @ W2 + b2)                       # output
    loss = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # BACKWARD: galti ko peeche ki taraf bhejo
    d_out = p - y                                  # output ki galti
    dW2 = h.T @ d_out / len(X)
    db2 = d_out.mean(axis=0, keepdims=True)
    d_h = (d_out @ W2.T) * h * (1 - h)             # galti hidden layer tak pahunchi
    dW1 = X.T @ d_h / len(X)
    db1 = d_h.mean(axis=0, keepdims=True)

    # UPDATE
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if step % 1000 == 0:
        print("step", step, "| loss", round(float(loss), 4))

print("predictions:", p.round(2).ravel())