import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# ── HOUR 4: BIAS VARIANCE + MULTIPLE REGRESSION ───────────────

print("=" * 50)
print("BIAS vs VARIANCE — UNDERFITTING vs OVERFITTING")
print("=" * 50)

# Create data that follows a curve
np.random.seed(42)
x = np.linspace(0, 10, 30)
y = 2 * x + np.sin(x) * 3 + np.random.normal(0, 1, 30)  # slight curve + noise
x = x.reshape(-1, 1)

# Split into train and test
split = 20
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# ── UNDERFITTING: degree 1 (too simple) ───────────────────────
poly1 = PolynomialFeatures(degree=1)
x_train_p1 = poly1.fit_transform(x_train)
x_test_p1  = poly1.transform(x_test)

model1 = LinearRegression()
model1.fit(x_train_p1, y_train)
train_score1 = model1.score(x_train_p1, y_train)
test_score1  = model1.score(x_test_p1, y_test)
print(f"\nUNDERFITTING (degree 1 — too simple):")
print(f"  Train accuracy: {train_score1:.3f}")
print(f"  Test accuracy:  {test_score1:.3f}")

# ── GOOD FIT: degree 3 ────────────────────────────────────────
poly3 = PolynomialFeatures(degree=3)
x_train_p3 = poly3.fit_transform(x_train)
x_test_p3  = poly3.transform(x_test)

model3 = LinearRegression()
model3.fit(x_train_p3, y_train)
train_score3 = model3.score(x_train_p3, y_train)
test_score3  = model3.score(x_test_p3, y_test)
print(f"\nGOOD FIT (degree 3 — just right):")
print(f"  Train accuracy: {train_score3:.3f}")
print(f"  Test accuracy:  {test_score3:.3f}")

# ── OVERFITTING: degree 15 (too complex) ──────────────────────
poly15 = PolynomialFeatures(degree=15)
x_train_p15 = poly15.fit_transform(x_train)
x_test_p15  = poly15.transform(x_test)

model15 = LinearRegression()
model15.fit(x_train_p15, y_train)
train_score15 = model15.score(x_train_p15, y_train)
test_score15  = model15.score(x_test_p15, y_test)
print(f"\nOVERFITTING (degree 15 — too complex):")
print(f"  Train accuracy: {train_score15:.3f}")
print(f"  Test accuracy:  {test_score15:.3f}")

# ── VISUALIZE ALL THREE ────────────────────────────────────────
x_plot = np.linspace(0, 10, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Underfitting\n(High Bias)',
          'Just Right\n(Good Fit)',
          'Overfitting\n(High Variance)']
models = [model1, model3, model15]
polys  = [poly1,  poly3,  poly15]
colors = ['#f43f5e', '#34d399', '#a78bfa']

for ax, title, model, poly, color in zip(axes, titles, models, polys, colors):
    x_plot_p = poly.transform(x_plot)
    y_plot   = model.predict(x_plot_p)

    ax.scatter(x_train, y_train, color='white', s=40, alpha=0.7, label='Train')
    ax.scatter(x_test,  y_test,  color='#fbbf24', s=40, alpha=0.9, label='Test')
    ax.plot(x_plot, y_plot, color=color, linewidth=2.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_facecolor('#0f0f1c')
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 30)

fig.patch.set_facecolor('#07070f')
plt.suptitle('Bias vs Variance Trade-off', fontsize=14,
             fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('hour4_bias_variance.png', dpi=150)
plt.show()
print("\nChart saved: hour4_bias_variance.png")

# ── MULTIPLE LINEAR REGRESSION ────────────────────────────────
print("\n" + "=" * 50)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 50)

# Cricket data: features = [practice_hours, matches_played, age]
# Target = runs scored
X_cricket = np.array([
    [1, 5,  20],
    [2, 10, 21],
    [3, 15, 22],
    [4, 20, 23],
    [5, 25, 24],
    [6, 30, 25],
])
y_cricket = np.array([20, 40, 58, 78, 95, 115])

# Train model
model_multi = LinearRegression()
model_multi.fit(X_cricket, y_cricket)

print(f"\nWeights learned:")
features = ['practice_hours', 'matches_played', 'age']
for feature, weight in zip(features, model_multi.coef_):
    print(f"  {feature:20} w = {weight:.4f}")
print(f"  {'bias':20} b = {model_multi.intercept_:.4f}")

# Predict for a new player
new_player = np.array([[7, 35, 26]])
prediction = model_multi.predict(new_player)[0]
print(f"\nNew player (7hrs, 35 matches, age 26):")
print(f"Predicted runs = {prediction:.1f}")

print("\n" + "=" * 50)
print("KEY CONCEPTS — WRITE IN NOTEBOOK")
print("=" * 50)
print("Underfitting = model too simple = high bias")
print("  Sign: low train accuracy AND low test accuracy")
print("Overfitting  = model too complex = high variance")
print("  Sign: high train accuracy BUT low test accuracy")
print("Goal: high accuracy on BOTH train and test")
print("\nMultiple features: f(x) = w1*x1 + w2*x2 + w3*x3 + b")
print("Each feature gets its own weight learned by gradient descent")