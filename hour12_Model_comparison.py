import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

print("=" * 50)
print("MODEL COMPARISON — ALL COURSE 1 ALGORITHMS")
print("=" * 50)

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         GaussianNB(),
    "KNN (K=5)":           KNeighborsClassifier(n_neighbors=5),
}

print(f"\n{'Model':<25} {'Train':>8} {'Test':>8} "
      f"{'CV Mean':>10} {'CV Std':>8}")
print("-" * 62)

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    train_acc = model.score(X_train_s, y_train)
    test_acc  = model.score(X_test_s, y_test)
    cv        = cross_val_score(model, X_train_s, y_train, cv=5)
    results[name] = {
        "train": train_acc, "test": test_acc,
        "cv_mean": cv.mean(), "cv_std": cv.std()
    }
    print(f"{name:<25} {train_acc:>8.4f} {test_acc:>8.4f} "
          f"{cv.mean():>10.4f} {cv.std():>8.4f}")

best = max(results, key=lambda k: results[k]["cv_mean"])
print(f"\nBest model by CV score: {best}")

# Visualize
names       = list(results.keys())
train_scores = [results[n]["train"]   for n in names]
test_scores  = [results[n]["test"]    for n in names]
cv_scores    = [results[n]["cv_mean"] for n in names]

x = np.arange(len(names))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, train_scores, w, label='Train',   color='#a78bfa', alpha=0.8)
ax.bar(x,     test_scores,  w, label='Test',    color='#38bdf8', alpha=0.8)
ax.bar(x + w, cv_scores,    w, label='CV Mean', color='#34d399', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('Accuracy')
ax.set_ylim(0.85, 1.01)
ax.set_title('Model Comparison — Course 1 Algorithms',
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('hour12_model_comparison.png', dpi=150)
plt.show()
print("Chart saved: hour12_model_comparison.png")