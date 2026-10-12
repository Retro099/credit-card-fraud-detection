import joblib
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "fraud_model_v1.joblib"
model = joblib.load(MODEL_PATH)
THRESHOLD = 0.5

# Exact column order used during training
COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def predict_fraud(features: list) -> dict:
    # Convert list to DataFrame with correct column names
    if len(features) != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} features, got {len(features)}")
    df_input = pd.DataFrame([features], columns=COLUMNS)
    
    prob = model.predict_proba(df_input)[0][1]
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(prob > THRESHOLD),
        "recommendation": "🚨 BLOCK TRANSACTION" if prob > THRESHOLD else "✅ Normal"
    }

# Test
if __name__ == "__main__":
    sample = [0, 100] + [0.0] * 28   # dummy input
    print(predict_fraud(sample))