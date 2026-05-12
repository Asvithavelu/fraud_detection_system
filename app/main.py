from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="Fraud Detection System")

MODEL_PATH = "model/fraud_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run train_model.py first")

model = joblib.load(MODEL_PATH)

# -------------------------------
# Input schema
# -------------------------------

class Transaction(BaseModel):
    amount: float
    transaction_time: int
    old_balance: float
    new_balance: float
    transaction_type: int
    device_type: int
    location_risk: int
    failed_logins: int
    is_foreign_transaction: int

# -------------------------------
# Home route
# -------------------------------

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
    }

# -------------------------------
# Prediction route
# -------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    try:

        data = np.array([[
            transaction.amount,
            transaction.transaction_time,
            transaction.old_balance,
            transaction.new_balance,
            transaction.transaction_type,
            transaction.device_type,
            transaction.location_risk,
            transaction.failed_logins,
            transaction.is_foreign_transaction
        ]])

        probability = model.predict_proba(data)[0][1]

        if probability >= 0.70:
            status = "Fraud"
            alert = "High-risk transaction detected"

        elif probability >= 0.40:
            status = "Suspicious"
            alert = "Manual review required"

        else:
            status = "Legitimate"
            alert = "Transaction is safe"

        return {
            "transaction_status": status,
            "risk_score": round(float(probability), 2),
            "alert": alert
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))