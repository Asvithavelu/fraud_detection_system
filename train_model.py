import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# -------------------------------
# Generate synthetic dataset
# -------------------------------

np.random.seed(42)
n = 5000

data = pd.DataFrame({
    "amount": np.random.randint(100, 100000, n),
    "transaction_time": np.random.randint(0, 24, n),
    "old_balance": np.random.randint(1000, 200000, n),
    "new_balance": np.random.randint(0, 200000, n),
    "transaction_type": np.random.randint(0, 4, n),
    "device_type": np.random.randint(0, 3, n),
    "location_risk": np.random.randint(0, 10, n),
    "failed_logins": np.random.randint(0, 5, n),
    "is_foreign_transaction": np.random.randint(0, 2, n)
})

# -------------------------------
# Create fraud labels
# -------------------------------

balance_difference = data["old_balance"] - data["new_balance"]

fraud_probability = (
    (data["amount"] / 100000) * 0.25 +
    (data["transaction_time"] / 24) * 0.15 +
    (balance_difference / 200000) * 0.20 +
    (data["location_risk"] / 10) * 0.15 +
    (data["failed_logins"] / 5) * 0.15 +
    (data["is_foreign_transaction"]) * 0.10
)

random_noise = np.random.rand(n) * 0.35

data["is_fraud"] = (
    fraud_probability + random_noise > 0.65
).astype(int)

# -------------------------------
# Split features and target
# -------------------------------

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

print("Fraud class count:")
print(y.value_counts())

# -------------------------------
# Train-test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Build model
# -------------------------------

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

# -------------------------------
# Train model
# -------------------------------

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Evaluation
# -------------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -------------------------------
# Cross validation
# -------------------------------

cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")

print("\nCross Validation F1 Scores:")
print(cv_scores)

print("\nAverage CV F1 Score:", cv_scores.mean())

# -------------------------------
# Save model
# -------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fraud_model.pkl")

print("\nModel saved successfully.")