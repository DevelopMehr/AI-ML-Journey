import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("XGBOOST — EXTREME GRADIENT BOOSTING")
print("=" * 50)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# DEFAULT XGBOOST
xgb = XGBClassifier(random_state=42,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train_s, y_train)
xgb_cv = cross_val_score(xgb, X_train_s, y_train, cv=5)

print(f"\nDefault XGBoost:")
print(f"  Train: {xgb.score(X_train_s, y_train):.4f}")
print(f"  Test:  {xgb.score(X_test_s, y_test):.4f}")
print(f"  CV:    {xgb_cv.mean():.4f}")

# TUNED XGBOOST
xgb_tuned = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_tuned.fit(X_train_s, y_train)
xgb_tuned_cv = cross_val_score(
    xgb_tuned, X_train_s, y_train, cv=5)

print(f"\nTuned XGBoost:")
print(f"  Train: {xgb_tuned.score(X_train_s, y_train):.4f}")
print(f"  Test:  {xgb_tuned.score(X_test_s, y_test):.4f}")
print(f"  CV:    {xgb_tuned_cv.mean():.4f}")

# RANDOM FOREST COMPARISON
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)
rf_cv = cross_val_score(rf, X_train_s, y_train, cv=5)

print(f"\nRandom Forest (comparison):")
print(f"  Train: {rf.score(X_train_s, y_train):.4f}")
print(f"  Test:  {rf.score(X_test_s, y_test):.4f}")
print(f"  CV:    {rf_cv.mean():.4f}")

winner = ('XGBoost Tuned' if xgb_tuned_cv.mean()
          > rf_cv.mean() else 'Random Forest')
print(f"\nWinner by CV score: {winner}")

# EFFECT OF LEARNING RATE
print(f"\nEffect of learning_rate on CV score:")
for lr in [0.001, 0.01, 0.05, 0.1, 0.3, 1.0]:
    m = XGBClassifier(learning_rate=lr, n_estimators=100,
                      random_state=42,
                      eval_metric='logloss', verbosity=0)
    cv = cross_val_score(m, X_train_s, y_train, cv=5)
    bar = "█" * int(cv.mean() * 50)
    print(f"  lr={lr:5.3f}  CV={cv.mean():.4f}  {bar}")

print("\nKEY RULES:")
print("Lower learning_rate + more trees = better accuracy")
print("subsample + colsample = prevent overfitting")
print("XGBoost wins most Kaggle tabular competitions")