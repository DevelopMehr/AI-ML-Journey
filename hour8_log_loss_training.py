import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("LOGISTIC REGRESSION — FULL TRAINING FROM SCRATCH")
print("=" * 50)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def log_loss_cost(X, y, w, b):
    m = len(y)
    f = sigmoid(X @ w + b)
    eps = 1e-15
    cost = -(1/m) * np.sum(y * np.log(f + eps) + (1-y) * np.log(1-f + eps))
    return cost

def compute_gradient_logistic(X, y, w, b):
    m = len(y)
    f = sigmoid(X @ w + b)
    error = f - y
    dj_dw = (1/m) * X.T @ error
    dj_db = (1/m) * np.sum(error)
    return dj_dw, dj_db

def train_logistic(X, y, alpha=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % 100 == 0:
            cost = log_loss_cost(X, y, w, b)
            cost_history.append(cost)
    return w, b, cost_history

X, y = make_classification(n_samples=200, n_features=2,
                             n_informative=2, n_redundant=0,
                             random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

w, b, costs = train_logistic(X_train_s, y_train, alpha=1, iterations=1000)

def predict(X, w, b):
    probs = sigmoid(X @ w + b)
    return (probs >= 0.5).astype(int)

y_pred  = predict(X_test_s, w, b)
accuracy = np.mean(y_pred == y_test)

print(f"\nTraining complete:")
print(f"  Final cost:    {costs[-1]:.4f}")
print(f"  Test accuracy: {accuracy:.4f}")
print(f"  Learned weights: {w}")
print(f"  Learned bias:    {b:.4f}")

from sklearn.linear_model import LogisticRegression
sk_model = LogisticRegression()
sk_model.fit(X_train_s, y_train)
print(f"\nSklearn accuracy: {sk_model.score(X_test_s, y_test):.4f}")
print("Both should be similar — confirms scratch implementation works")