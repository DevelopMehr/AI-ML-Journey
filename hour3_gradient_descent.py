import numpy as np
import matplotlib.pyplot as plt

# ── HOUR 3: COST FUNCTION + GRADIENT DESCENT ──────────────────

print("=" * 50)
print("COST FUNCTION + GRADIENT DESCENT")
print("=" * 50)

# Training data
x_train = np.array([1, 2, 3, 4, 5], dtype=float)
y_train = np.array([20, 35, 50, 65, 80], dtype=float)
m = len(x_train)

# ── COST FUNCTION ──────────────────────────────────────────────
def compute_cost(x, y, w, b):
    """
    Calculate how wrong the model is.
    Lower cost = better model.
    """
    m = len(x)
    predictions = w * x + b          # all predictions at once
    errors = predictions - y          # all errors at once
    squared_errors = errors ** 2      # square all errors
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost

# Test cost with different w values
print("\nCost with different values of w (b=5 fixed):")
for w_test in [5, 10, 15, 20, 25]:
    cost = compute_cost(x_train, y_train, w_test, 5)
    bar = "█" * int(cost / 50)
    print(f"  w={w_test:2.0f}  Cost={cost:8.2f}  {bar}")

print(f"\nBest w is 15 — lowest cost confirms our manual choice was correct")

# ── GRADIENT DESCENT ───────────────────────────────────────────
def compute_gradient(x, y, w, b):
    """
    Calculate which direction to move w and b
    to reduce cost.
    """
    m = len(x)
    predictions = w * x + b
    errors = predictions - y

    dj_dw = (1 / m) * np.sum(errors * x)   # gradient for w
    dj_db = (1 / m) * np.sum(errors)        # gradient for b
    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, num_iterations):
    """
    Run gradient descent to find best w and b.
    """
    w = w_init
    b = b_init
    cost_history = []

    for i in range(num_iterations):
        # Calculate gradients
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update w and b — move downhill
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost every 100 steps
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            cost_history.append(cost)

    return w, b, cost_history

# Run gradient descent
print("\nRunning Gradient Descent...")
w_final, b_final, cost_history = gradient_descent(
    x_train, y_train,
    w_init=0, b_init=0,    # start from zero
    alpha=0.01,             # learning rate
    num_iterations=2000
)

print(f"Starting point: w=0, b=0")
print(f"Final w = {w_final:.4f}")
print(f"Final b = {b_final:.4f}")
print(f"Expected: w≈15, b≈5")
print(f"Final cost = {compute_cost(x_train, y_train, w_final, b_final):.6f}")

# Make predictions with learned parameters
print("\nPredictions with learned model:")
for x, y_actual in zip(x_train, y_train):
    pred = w_final * x + b_final
    print(f"  Hours={x:.0f}  Predicted={pred:.1f}  Actual={y_actual:.1f}")

# Predict new value
new_hours = 6
print(f"\nFor {new_hours} hours: predicted runs = {w_final * new_hours + b_final:.1f}")

# ── VISUALIZE COST DECREASING ──────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(len(cost_history)), cost_history, color='#f43f5e', linewidth=2)
plt.title('Cost Decreasing During Gradient Descent', fontsize=13, fontweight='bold')
plt.xlabel('Iterations (×100)')
plt.ylabel('Cost J(w,b)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hour3_cost_decreasing.png', dpi=150)
plt.show()
print("Chart saved: hour3_cost_decreasing.png")

print("\n" + "=" * 50)
print("KEY FORMULAS TO MEMORIZE")
print("=" * 50)
print("Model:    f(x) = wx + b")
print("Cost:     J(w,b) = (1/2m) × SUM[(f(xi)-yi)²]")
print("Update w: w = w - α × dJ/dw")
print("Update b: b = b - α × dJ/db")