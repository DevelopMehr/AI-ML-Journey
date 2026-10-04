import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("NEURAL NETWORK TRAINING WITH TENSORFLOW")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")

# LOAD AND PREPARE DATA
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nDataset: Breast Cancer Detection")
print(f"Training: {len(X_train)} samples")
print(f"Test:     {len(X_test)} samples")
print(f"Features: {X_train.shape[1]}")

# BUILD NEURAL NETWORK
model = Sequential([
    Dense(64, activation='relu',
          input_shape=(X_train_s.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])

model.summary()

# COMPILE
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# TRAIN
print("\nTraining neural network...")
history = model.fit(
    X_train_s, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
print("Training complete.")

# EVALUATE
loss, accuracy = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\nTest Results:")
print(f"  Test Loss:      {loss:.4f}")
print(f"  Test Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Final Train Acc: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Val Acc:   {history.history['val_accuracy'][-1]:.4f}")

# VISUALIZE TRAINING HISTORY
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],
             color='#a78bfa', linewidth=2, label='Train')
axes[0].plot(history.history['val_accuracy'],
             color='#34d399', linewidth=2, label='Validation')
axes[0].set_title('Accuracy During Training',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],
             color='#f43f5e', linewidth=2, label='Train')
axes[1].plot(history.history['val_loss'],
             color='#fbbf24', linewidth=2, label='Validation')
axes[1].set_title('Loss During Training',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hour19_training_history.png', dpi=150)
plt.show()
print("Chart saved: hour19_training_history.png")

print("\nBACKPROPAGATION CONCEPT:")
print("Forward:  input → layers → prediction")
print("Loss:     measure how wrong prediction was")
print("Backward: calculate gradient for every weight")
print("Update:   w = w - lr × gradient")
print("TensorFlow does backward pass automatically")
print("\nWHY TRAINING FAILS:")
print("Vanishing gradient → use ReLU not sigmoid in hidden layers")
print("Overfitting        → add Dropout or reduce neurons")
print("Wrong lr           → start 0.001, Adam adapts automatically")