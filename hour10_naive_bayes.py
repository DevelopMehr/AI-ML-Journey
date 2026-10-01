import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

print("=" * 50)
print("NAIVE BAYES — FULL IMPLEMENTATION")
print("=" * 50)


# ── BAYES THEOREM MANUAL EXAMPLE ──────────────────────────────

print("\nManual Bayes Theorem — Spam Detection Example:")
print("-" * 45)

p_spam = 0.30
p_word_given_spam = 0.80
p_word_given_ham = 0.10

# P(word)
p_word = (
    p_word_given_spam * p_spam
    + p_word_given_ham * (1 - p_spam)
)

# P(spam|word)
p_spam_given_word = (
    p_word_given_spam * p_spam
) / p_word

print(f"P(spam)              = {p_spam}")
print(f"P('offer'|spam)      = {p_word_given_spam}")
print(f"P('offer'|not spam)  = {p_word_given_ham}")
print(f"P('offer')           = {p_word:.3f}")
print(
    f"P(spam|'offer')      = "
    f"{p_spam_given_word:.3f} "
    f"({p_spam_given_word * 100:.0f}%)"
)


# ── GAUSSIAN NAIVE BAYES ON IRIS ──────────────────────────────

print("\n" + "=" * 50)
print("GAUSSIAN NAIVE BAYES — IRIS DATASET")
print("=" * 50)

data = load_iris()

X, y = data.data, data.target

print(f"\nFeatures: {data.feature_names}")
print(f"Classes:  {data.target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print(
    f"\nGaussian NB Accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}"
)


# What the model learned

print("\nClass means (what NB thinks each class looks like):")

for i, class_name in enumerate(data.target_names):

    means = gnb.theta_[i]

    print(
        f"  {class_name}: "
        f"sepal_len={means[0]:.2f}, "
        f"sepal_wid={means[1]:.2f}, "
        f"petal_len={means[2]:.2f}, "
        f"petal_wid={means[3]:.2f}"
    )


# Cross-validation

scores = cross_val_score(gnb, X, y, cv=5)

print(
    f"\n5-fold cross-validation scores: "
    f"{scores.round(3)}"
)

print(
    f"Mean accuracy: "
    f"{scores.mean():.4f} ± {scores.std():.4f}"
)


# Predict one new flower

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

probs = gnb.predict_proba(new_flower)[0]

pred = gnb.predict(new_flower)[0]

print("\nNew flower prediction:")

for name, prob in zip(data.target_names, probs):

    print(
        f"  P({name}) = {prob:.4f}"
    )

print(
    f"  Predicted class: "
    f"{data.target_names[pred]}"
)


# ── KEY POINTS ────────────────────────────────────────────────

print("\n" + "=" * 50)
print("KEY POINTS — WRITE IN NOTEBOOK")
print("=" * 50)

print("P(A|B) = P(B|A) × P(A) / P(B)")
print("Naive = assumes all features are independent")
print("Works great for text classification despite wrong assumption")
print("Gaussian NB: assumes features follow normal distribution")