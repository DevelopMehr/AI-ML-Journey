import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("CLUSTERING — UNSUPERVISED LEARNING")
print("=" * 50)

# K-MEANS + ELBOW METHOD
X_blobs, true_labels = make_blobs(
    n_samples=300, centers=4,
    random_state=42, cluster_std=0.8)

print("\nElbow Method — finding best K:")
inertias = []
k_range  = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_blobs)
    inertias.append(km.inertia_)
    print(f"  K={k:2d}  Inertia={km.inertia_:10.2f}")

# Plot all three side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Elbow curve
axes[0].plot(list(k_range), inertias, marker='o',
             color='#a78bfa', linewidth=2.5)
axes[0].axvline(4, color='#f43f5e', linewidth=2,
                linestyle='--', label='Elbow K=4')
axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: