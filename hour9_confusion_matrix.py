import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score,
                              classification_report)

print("=" * 50)
print("CONFUSION MATRIX + EVALUATION METRICS")
print("=" * 50)

data = load_breast_cancer()
X, y = data.data, data.target
print(f"\nDataset: Breast Cancer Detection")
print(f"Features: {X.shape[1]}")
print(f"Samples:  {X.shape[0]}")
print(f"Classes:  {data.target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"  True Positives  (caught cancer):     {tp}")
print(f"  True Negatives  (correctly cleared): {tn}")
print(f"  False Positives (false alarm):        {fp}")
print(f"  False Negatives (missed cancer):      {fn}")

print(f"\nFull report:")
print(classification_report(y_test, y_pred,
      target_names=data.target_names))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            ax=ax, linewidths=2)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label', fontsize=12)
ax.set_title(f'Confusion Matrix — Accuracy: {accuracy*100:.1f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hour9_confusion_matrix.png', dpi=150)
plt.show()
print("Chart saved: hour9_confusion_matrix.png")