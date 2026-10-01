import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

print("=" * 50)
print("RIDGE AND LASSO REGULARIZATION")
print("=" * 50)

# 20 features but only 5 actually matter
# Perfect dataset to test regularization
X, y = make_regression(n_samples=100, n_features=20,
                        n_informative=5,
                        noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# NO REGULARIZATION
lr = LinearRegression()
lr.fit(X_train_s, y_train)
print(f"\nNo Regularization:")
print(f"  Train R2: {lr.score(X_train_s, y_train):.4f}")
print(f"  Test  R2: {lr.score(X_test_s,  y_test):.4f}")
print(f"  Largest weight: {max(abs(lr.coef_)):.2f}")

# RIDGE L2
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
print(f"\nRidge L2 alpha=1.0:")
print(f"  Train R2: {ridge.score(X_train_s, y_train):.4f}")
print(f"  Test  R2: {ridge.score(X_test_s,  y_test):.4f}")
print(f"  Largest weight: {max(abs(ridge.coef_)):.2f}")
print(f"  Weights at zero: {sum(ridge.coef_ == 0)}")

# LASSO L1
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_s, y_train)
print(f"\nLasso L1 alpha=0.1:")
print(f"  Train R2: {lasso.score(X_train_s, y_train):.4f}")
print(f"  Test  R2: {lasso.score(X_test_s,  y_test):.4f}")
print(f"  Largest weight: {max(abs(lasso.coef_)):.2f}")
print(f"  Weights at zero: {sum(lasso.coef_ == 0)}")

# WHICH FEATURES DID LASSO KEEP
print(f"\nLasso feature selection:")
for i, w in enumerate(lasso.coef_):
    status = "KEPT   " if w != 0 else "REMOVED"
    print(f"  Feature {i:2d}: {status}  weight={w:.3f}")

# EFFECT OF DIFFERENT LAMBDA VALUES
print(f"\nHow lambda affects Ridge test score:")
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    m = Ridge(alpha=alpha)
    m.fit(X_train_s, y_train)
    print(f"  lambda={alpha:6.3f}  Test R2={m.score(X_test_s, y_test):.4f}")