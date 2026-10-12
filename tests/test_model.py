import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import COLUMNS, model, predict_fraud  # noqa: E402


def test_model_loaded():
    assert model is not None, "Model failed to load"


def test_prediction_shape():
    rng = np.random.default_rng(0)
    df_input = pd.DataFrame([rng.standard_normal(30).tolist()], columns=COLUMNS)
    prob = float(model.predict_proba(df_input)[0][1])
    assert 0 <= prob <= 1


def test_column_order():
    assert COLUMNS[:2] == ["Time", "Amount"] and COLUMNS[-1] == "V28" and len(COLUMNS) == 30


def test_predict_fraud_output_keys():
    out = predict_fraud([0, 100] + [0.0] * 28)
    assert set(out) == {"fraud_probability", "is_fraud", "recommendation"}
    assert out["is_fraud"] == (out["fraud_probability"] > 0.5)


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        predict_fraud([0.0] * 5)


def test_extreme_v14_raises_score():
    base = [0, 100] + [0.0] * 28
    low = list(base)
    low[COLUMNS.index("V14")] = -15.0
    assert predict_fraud(low)["fraud_probability"] >= predict_fraud(base)["fraud_probability"]
