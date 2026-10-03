import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

print("=" * 50)
print("DECISION TREES")
print("=" * 50)

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# EFFECT OF max_depth
print("\nEffect of max_depth on train vs test accuracy:")
print(f"{'Depth':>10} {'Train':>8} {'Test':>8} {'CV':>8} {'Status':>12}")
print("-" * 52)

depths = [1, 2, 3, 4, 5, 6, None]
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = dt.score(X_train, y_train)
    test_acc  = dt.score(X_test, y_test)
    cv        = cross_val_score(dt, X, y, cv=5).mean()

    if train_acc - test_acc > 0.1:
        status = "OVERFITTING"
    elif train_acc < 0.85:
        status = "UNDERFIT"
    else:
        status = "good"

    d_str = str(depth) if depth else "None(full)"
    print(f"{d_str:>10} {train_acc:>8.4f} {test_acc:>8.4f} "
          f"{cv:>8.4f} {status:>12}")

# BEST DEPTH
best_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
best_dt.fit(X_train, y_train)

print(f"\nDecision rules (max_depth=3):")
print(export_text(best_dt, feature_names=list(data.feature_names)))

print(f"\nFeature importances:")
for name, imp in zip(data.feature_names, best_dt.feature_importances_):
    bar = "█" * int(imp * 30)
    print(f"  {name:25} {imp:.4f}  {bar}")

print(f"\nTest accuracy: {best_dt.score(X_test, y_test):.4f}")

print("\nKEY RULES:")
print("Gini = 1 - SUM(pi^2)")
print("Lower Gini = purer node = better split")
print("max_depth too large = overfitting")
print("max_depth too small = underfitting")
print("No scaling needed for decision trees")