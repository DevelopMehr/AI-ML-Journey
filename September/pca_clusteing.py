import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans

print("=" * 50)
print("PCA — DIMENSIONALITY REDUCTION")
print("=" * 50)

data = load_breast_cancer()
X, y = data.data, data.target

print(f"\nOriginal shape: {X.shape}")
print(f"Features: {X.shape[1]}")

# Always scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# FIT PCA — keep all components first
pca_full = PCA()
pca_full.fit(X_scaled)

# Explained variance
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance:")
for i, var in enumerate(cumulative[:10], 1):
    bar = "█" * int(var * 30)
    print(f"  {i:2d} components: {var:.4f}  {bar}")

# Find how many components explain 95% variance
n_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_95}")

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print(f"\nAfter PCA: {X_2d.shape}")
print(f"PC1 explains: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"PC2 explains: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"Total kept:   {sum(pca_2d.explained_variance_ratio_):.4f}")

# Visualize in 2D
plt.figure(figsize=(8, 6))
colors = ['#f43f5e', '#34d399']
labels = ['Malignant', 'Benign']
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=color, label=label, s=30, alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Breast Cancer Data — Visualized in 2D with PCA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=150)
plt.show()
print("Saved: pca_visualization.png")

# PCA + K-Means pipeline
print("\n" + "=" * 50)
print("PCA + K-MEANS PIPELINE")
print("=" * 50)

pca_km = PCA(n_components=n_95)
X_reduced = pca_km.fit_transform(X_scaled)
print(f"Reduced from {X.shape[1]} to {n_95} features")

km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km.fit_predict(X_reduced)

# Compare with true labels
match = np.mean(labels_km == y)
match = max(match, 1 - match)  # handle label flip
print(f"Cluster-label agreement: {match:.4f}")

print("\nKEY RULES:")
print("Always StandardScaler before PCA")
print("Choose n_components to explain 95% variance")
print("PCA for visualization: n_components=2")
print("PCA + clustering: reduce first then cluster")