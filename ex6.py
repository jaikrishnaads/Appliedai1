# Experiment 6: Optimization Techniques in DNN

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Dataset (Update path as needed)
# df = pd.read_csv(r"Z:\ai lab\bank-full.csv", sep=';') 
# For demonstration, assuming df is loaded. 
# Please ensure the file path is correct on your machine.
try:
    df = pd.read_csv(r"bank-full.csv", sep=';')
except FileNotFoundError:
    print("Please update the file path to bank-full.csv")
    # Stop execution if file not found to prevent errors
    raise FileNotFoundError("Bank dataset not found. Please update the path.")

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nMissing values before preprocessing:\n{df.isnull().sum()}")

# Handle missing values (replace 'unknown' with NaN and forward fill)
df.replace("unknown", np.nan, inplace=True)
df.ffill(inplace=True)

# Encode target variable
df['y'] = df['y'].map({'yes': 1, 'no': 0})
print(f"\nTarget distribution:\n{df['y'].value_counts()}")

# Separate features and target
X = df.drop('y', axis=1)
y = df['y']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
print(f"\nFeature shape after encoding: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Split temp: 50% validation, 50% test (15% each of original)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Features standardized successfully!")

def build_model(input_dim, optimizer):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

optimizers = {
    "Batch GD": SGD(learning_rate=0.01),
    "SGD": SGD(learning_rate=0.01),
    "Mini-Batch": SGD(learning_rate=0.01),
    "Momentum": SGD(learning_rate=0.01, momentum=0.9),
    "Nesterov": SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
}

batch_sizes = {
    "Batch GD": len(X_train),
    "SGD": 1,
    "Mini-Batch": 32,
    "Momentum": 32,
    "Nesterov": 32
}

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

histories = {}
times = {}
models = {}

# Train each optimizer
for name, optimizer in optimizers.items():
    print(f"\n{'='*60}")
    print(f"Training with {name}")
    print(f"{'='*60}")
    model = build_model(X_train.shape[1], optimizer)
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=batch_sizes[name],
        callbacks=[early_stop],
        verbose=1
    )
    end = time.time()
    histories[name] = history
    times[name] = end - start
    models[name] = model
    print(f"\nTraining completed in {times[name]:.2f} seconds")

# Plotting Results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for name, history in histories.items():
    axes[0].plot(history.history['loss'], label=name, linewidth=2)
    axes[0].set_title("Training Loss vs Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
for name, history in histories.items():
    axes[1].plot(history.history['val_loss'], label=name, linewidth=2)
    axes[1].set_title("Validation Loss vs Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Select Best Optimizer
val_accuracies = {}
for name, history in histories.items():
    final_val_acc = history.history['val_accuracy'][-1]
    val_accuracies[name] = final_val_acc

best_optimizer_name = max(val_accuracies, key=val_accuracies.get)
best_model = models[best_optimizer_name]
print(f"Best Optimizer: {best_optimizer_name}")

# Evaluate on Test Set
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
