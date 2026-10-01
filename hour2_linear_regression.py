import numpy as np
import matplotlib.pyplot as plt

# ── HOUR 2: LINEAR REGRESSION ─────────────────────────────────

print("=" * 50)
print("LINEAR REGRESSION — f(x) = wx + b")
print("=" * 50)

# Our training data
# x = practice hours, y = runs scored
x_train = np.array([1, 2, 3, 4, 5], dtype=float)
y_train = np.array([20, 35, 50, 65, 80], dtype=float)

# ── MANUAL PREDICTION ──────────────────────────────────────────
# We manually set w and b first to understand the concept
w = 15.0   # slope — each hour adds 15 runs
b = 5.0    # starting point — 5 runs with zero practice

print(f"\nManual model: f(x) = {w}x + {b}")
print("\nPredictions vs Actual:")
for x, y_actual in zip(x_train, y_train):
    prediction = w * x + b
    error = y_actual - prediction
    print(f"  Hours={x:.0f}  Predicted={prediction:.1f}  Actual={y_actual:.1f}  Error={error:.1f}")

# Predict for new input
new_hours = 6
prediction = w * new_hours + b
print(f"\nFor {new_hours} hours of practice: predicted runs = {prediction:.1f}")

# ── VISUALIZE ──────────────────────────────────────────────────
x_line = np.linspace(0, 7, 100)
y_line = w * x_line + b

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color='#f43f5e', s=100,
            zorder=5, label='Training data')
plt.plot(x_line, y_line, color='#a78bfa', linewidth=2.5,
         label=f'f(x) = {w}x + {b}')
plt.scatter([new_hours], [prediction], color='#34d399',
            s=150, zorder=6, marker='*', label=f'Prediction: {prediction:.0f} runs')

plt.title('Linear Regression — Practice Hours vs Runs Scored',
          fontsize=13, fontweight='bold')
plt.xlabel('Practice Hours')
plt.ylabel('Runs Scored')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hour2_linear_regression.png', dpi=150)
plt.show()
print("\nChart saved: hour2_linear_regression.png")

# ── KEY INSIGHT ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
print(f"w = {w} means each extra hour adds {w} runs")
print(f"b = {b} means starting prediction is {b} runs")
print("The model learns the best w and b from data")
print("That learning process is called Gradient Descent")
print("Hour 3 teaches exactly how Gradient Descent works")