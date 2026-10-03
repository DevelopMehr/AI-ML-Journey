import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("NEURAL NETWORK — ACTIVATION FUNCTIONS")
print("=" * 50)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def linear(z):
    return z

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu_derivative(z):
    return (z > 0).astype(float)

# Test values
z_values = np.array([-5, -2, -1, 0, 1, 2, 5])
print(f"\n{'z':>6} {'sigmoid':>10} {'relu':>8} {'linear':>8}")
print("-" * 36)
for z in z_values:
    print(f"{z:>6.1f} {sigmoid(z):>10.4f} "
          f"{relu(z):>8.4f} {linear(z):>8.4f}")

# VISUALIZE
z = np.linspace(-6, 6, 300)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

configs = [
    ("Sigmoid", sigmoid, "#a78bfa"),
    ("ReLU",    relu,    "#34d399"),
    ("Linear",  linear,  "#38bdf8"),
]
for ax, (name, func, color) in zip(axes, configs):
    ax.plot(z, func(z), color=color, linewidth=2.5)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

plt.suptitle('Activation Functions Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('hour14_activations.png', dpi=150)
plt.show()
print("Chart saved: hour14_activations.png")

# VANISHING GRADIENT DEMO
print("\nVanishing Gradient Demonstration:")
print(f"\n{'z':>6} {'sigmoid_grad':>15} {'relu_grad':>12}")
print("-" * 35)
for z in [0.1, 1.0, 5.0, 10.0, 20.0]:
    print(f"{z:>6.1f} {sigmoid_derivative(z):>15.8f} "
          f"{relu_derivative(np.array([z]))[0]:>12.4f}")

print("\nAt z=10: sigmoid gradient ≈ 0.000045 — almost ZERO")
print("At z=10: relu gradient = 1.0 — stays constant")
print("This is why ReLU is used in hidden layers.")

# MANUAL FORWARD PASS
print("\n" + "=" * 50)
print("MANUAL FORWARD PASS — ONE HIDDEN LAYER")
print("=" * 50)

x  = np.array([0.5, 0.8])
W1 = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6]])
b1 = np.array([0.1, 0.1, 0.1])
W2 = np.array([0.7, 0.8, 0.9])
b2 = 0.1

z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

print(f"\nInput x: {x}")
print(f"z1 (before ReLU):   {z1.round(4)}")
print(f"a1 (after ReLU):    {a1.round(4)}")
print(f"z2 (before sigmoid): {z2:.4f}")
print(f"a2 (after sigmoid):  {a2:.4f}")
print(f"Prediction: {'class 1' if a2 >= 0.5 else 'class 0'}")

print("\nRULES TO MEMORIZE:")
print("Hidden layers   → ReLU")
print("Binary output   → Sigmoid")
print("Multi-class out → Softmax")
print("Regression out  → Linear")
print("Vanishing grad  → caused by sigmoid in deep layers")