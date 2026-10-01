import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("LOGISTIC REGRESSION + SIGMOID")
print("=" * 50)

# SIGMOID FUNCTION FROM SCRATCH
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Test sigmoid on key values
test_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
print("\nSigmoid outputs:")
for z in test_values:
    output = sigmoid(z)
    bar = "█" * int(output * 20)
    print(f"  sigmoid({z:4.0f}) = {output:.4f}  {bar}")

# Three values to memorize
print(f"\nsigmoid(0)    = {sigmoid(0):.1f}   always exactly 0.5")
print(f"sigmoid(1000) = {sigmoid(1000):.4f} approaches 1")
print(f"sigmoid(-1000)= {sigmoid(-1000):.4f} approaches 0")

# SPAM DETECTION EXAMPLE
print("\n" + "=" * 50)
print("SPAM DETECTION EXAMPLE")
print("=" * 50)

w = 0.8
b = -2.0

emails = [0, 1, 2, 3, 4, 5, 10]
print(f"\nw={w}, b={b}")
print("Suspicious words → probability → decision")
print("-" * 45)
for words in emails:
    z    = w * words + b
    prob = sigmoid(z)
    pred = "SPAM" if prob >= 0.5 else "not spam"
    print(f"  {words} words → z={z:.1f} → prob={prob:.3f} → {pred}")

print("\n" + "=" * 50)
print("FORMULA SUMMARY")
print("=" * 50)
print("Step 1: z    = w*x + b")
print("Step 2: f(x) = 1 / (1 + e^(-z))")
print("Step 3: if f(x) >= 0.5 → class 1")
print("        if f(x) <  0.5 → class 0")