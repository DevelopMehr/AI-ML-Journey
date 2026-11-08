import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # 4 customers, 2 inputs
y = np.array([0, 1, 1, 1])                       # asli jawab

w = np.zeros(2)      # weights shuru mein zero
b = 0.0              # bias
lr = 0.5             # learning rate: step ka size

for step in range(1001):                         # 1001 baar dohrao (epochs)
    p = sigmoid(X @ w + b)                       # 1. forward
    loss = np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))   # 2. loss

    error = p - y                                # 3. gradient
    dw = X.T @ error / len(y)
    db = error.mean()

    w = w - lr * dw                              # 4. update
    b = b - lr * db

    if step % 200 == 0:
        print("step", step, "| loss", round(float(loss), 4))

print("weights:", w.round(2), "| bias:", round(float(b), 2))
print("predictions:", sigmoid(X @ w + b).round(2))