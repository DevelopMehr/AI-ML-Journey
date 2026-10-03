import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, BaggingClassifier)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("ENSEMBLE METHODS — BAGGING AND BOOSTING")
print("=" * 50)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Single Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Bagging (100 trees)":  BaggingClassifier(n_estimators=100, random_state=42),
    "Random Forest (100)":  RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost (100)":       AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print(f"\n{'Model':<28} {'Train':>8} {'Test':>8} {'CV Mean':>10}")
print("-" * 58)

cv_scores_all = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    train_acc = model.score(X_train_s, y_train)
    test_acc  = model.score(X_test_s, y_test)
    cv        = cross_val_score(model, X_train_s, y_train, cv=5)
    cv_scores_all[name] = cv.mean()
    print(f"{name:<28} {train_acc:>8.4f} {test_acc:>8.4f} "
          f"{cv.mean():>10.4f}")

# Feature importance
rf = models["Random Forest (100)"]
print(f"\nTop 10 most important features (Random Forest):")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for rank, idx in enumerate(indices, 1):
    bar = "█" * int(importances[idx] * 100)
    print(f"  {rank:2d}. {data.feature_names[idx]:30} "
          f"{importances[idx]:.4f}  {bar}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 5))
names  = list(cv_scores_all.keys())
scores = list(cv_scores_all.values())
colors = ['#f43f5e','#a78bfa','#34d399','#38bdf8','#fbbf24']
bars   = ax.bar(names, scores, color=colors, width=0.5)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f"{score:.4f}", ha='center',
            fontsize=9, fontweight='bold')
ax.set_ylim(0.88, 1.01)
ax.set_ylabel('CV Accuracy')
ax.set_title('Ensemble Methods vs Single Tree',
             fontsize=13, fontweight='bold')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('hour15_ensemble_comparison.png', dpi=150)
plt.show()
print("Chart saved: hour15_ensemble_comparison.png")

print("\nKEY DIFFERENCES:")
print("Bagging:  parallel trees → reduces VARIANCE")
print("Boosting: sequential trees → reduces BIAS")
print("Random Forest = Bagging + random feature subsets")