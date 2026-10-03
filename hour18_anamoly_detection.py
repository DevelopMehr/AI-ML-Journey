import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("ANOMALY DETECTION — GAUSSIAN METHOD")
print("=" * 50)

np.random.seed(42)

# GENERATE NORMAL TRANSACTION DATA
n_normal = 300
amount_normal = np.random.normal(100, 20, n_normal)
freq_normal   = np.random.normal(5, 1, n_normal)
X_normal = np.column_stack([amount_normal, freq_normal])

# ADD FRAUD TRANSACTIONS
n_fraud = 10
amount_fraud = np.random.uniform(500, 2000, n_fraud)
freq_fraud   = np.random.uniform(50, 100, n_fraud)
X_fraud = np.column_stack([amount_fraud, freq_fraud])

X_all = np.vstack([X_normal, X_fraud])
y_all = np.array([0]*n_normal + [1]*n_fraud)

# FIT GAUSSIAN ON NORMAL DATA ONLY
mu     = np.mean(X_normal, axis=0)
sigma2 = np.var(X_normal, axis=0)

print(f"\nGaussian fit on {n_normal} normal transactions:")
print(f"  Amount:    mean={mu[0]:.2f}, variance={sigma2[0]:.2f}")
print(f"  Frequency: mean={mu[1]:.2f}, variance={sigma2[1]:.2f}")

def gaussian_probability(X, mu, sigma2):
    k           = len(mu)
    det         = np.prod(sigma2)
    coefficient = 1 / ((2 * np.pi)**(k/2) * np.sqrt(det))
    exponent    = -0.5 * np.sum((X - mu)**2 / sigma2, axis=1)
    return coefficient * np.exp(exponent)

# Calculate probabilities
probs       = gaussian_probability(X_all, mu, sigma2)
epsilon     = np.percentile(probs[:n_normal], 5)
predictions = (probs < epsilon).astype(int)

# Results
true_pos  = np.sum((predictions == 1) & (y_all == 1))
false_pos = np.sum((predictions == 1) & (y_all == 0))
false_neg = np.sum((predictions == 0) & (y_all == 1))

print(f"\nEpsilon threshold: {epsilon:.2e}")
print(f"Fraud caught:          {true_pos}/{n_fraud}")
print(f"Normal wrongly flagged: {false_pos}/{n_normal}")
print(f"Fraud missed:          {false_neg}/{n_fraud}")

# VISUALIZE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot
axes[0].scatter(X_normal[:, 0], X_normal[:, 1],
                c='#34d399', s=20, alpha=0.5, label='Normal')
axes[0].scatter(X_fraud[:, 0], X_fraud[:, 1],
                c='#f43f5e', s=80, marker='X', label='Fraud (True)')
flagged = np.where(predictions == 1)[0]
axes[0].scatter(X_all[flagged, 0], X_all[flagged, 1],
                s=150, facecolors='none', edgecolors='#fbbf24',
                linewidth=2, label='Flagged by model')
axes[0].set_xlabel('Transaction Amount')
axes[0].set_ylabel('Transactions per Hour')
axes[0].set_title('Anomaly Detection Results',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Probability distribution
axes[1].hist(probs[:n_normal], bins=30,
             color='#34d399', alpha=0.7, label='Normal')
axes[1].hist(probs[n_normal:], bins=10,
             color='#f43f5e', alpha=0.9, label='Fraud')
axes[1].axvline(epsilon, color='#fbbf24', linewidth=2,
                linestyle='--', label=f'epsilon={epsilon:.2e}')
axes[1].set_xlabel('Probability p(x)')
axes[1].set_ylabel('Count')
axes[1].set_title('Probability Distribution',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hour18_anomaly_detection.png', dpi=150)
plt.show()
print("Chart saved: hour18_anomaly_detection.png")

print("\nKEY RULES:")
print("Fit Gaussian on NORMAL data only")
print("Calculate p(x) for every new example")
print("p(x) < epsilon → anomaly")
print("Lower epsilon → fewer flags → higher precision")
print("Higher epsilon → more flags → higher recall")