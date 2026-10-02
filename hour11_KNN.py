import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("K-NEAREST NEIGHBORS")
print("=" * 50)

# MANUAL DISTANCE CALCULATION
print("\nManual Distance Calculation:")
point_A = np.array([2, 3])
point_B = np.array([5, 7])

euclidean = np.sqrt(np.sum((point_B - point_A)**2))
manhattan  = np.sum(np.abs(point_B - point_A))
print(f"Point A: {point_A}")
print(f"Point B: {point_B}")
print(f"Euclidean distance: {euclidean:.4f}")
print(f"Manhattan distance: {manhattan:.4f}")

# KNN ON IRIS
data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# MUST scale for KNN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# FIND BEST K
print("\nFinding best K using cross-validation:")
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5)
    cv_scores.append(scores.mean())
    print(f"  K={k:2d}  CV accuracy={scores.mean():.4f}")

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest K = {best_k}")
print(f"Best CV accuracy = {max(cv_scores):.4f}")

# Train with best K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_s, y_train)
test_acc = knn_best.score(X_test_s, y_test)
print(f"Test accuracy with K={best_k}: {test_acc:.4f}")

# VISUALIZE K vs ACCURACY
plt.figure(figsize=(9, 5))
plt.plot(list(k_values), cv_scores, marker='o',
         color='#a78bfa', linewidth=2.5)
plt.axvline(best_k, color='#f43f5e', linewidth=2,
            linestyle='--', label=f'Best K={best_k}')
plt.title('K vs Cross-Validation Accuracy',
          fontsize=13, fontweight='bold')
plt.xlabel('K (number of neighbors)')
plt.ylabel('CV Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hour11_knn_k_selection.png', dpi=150)
plt.show()
print("Chart saved: hour11_knn_k_selection.png")

print("\nKEY RULES:")
print("1. Always scale features before KNN")
print("2. Small K = overfitting")
print("3. Large K = underfitting")
print("4. Find best K with cross-validation")