"""Threshold / cost analysis on the Kaggle ULB credit-card data. CPU only, ~1-2 min.

Same split as the notebook (test_size=0.2, stratify, random_state=42).
The threshold is chosen on a VALIDATION split carved out of the training set
(25% of train, stratified, random_state=42), never on the test set.

usage: python scripts/threshold_analysis.py --csv data/creditcard.csv
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
PCA = [f"V{i}" for i in range(1, 29)]


def preprocessor():
    return ColumnTransformer([("scaler", StandardScaler(), ["Time", "Amount"]),
                              ("passthrough", "passthrough", PCA)], remainder="drop")


def xgb(spw):
    return Pipeline([("preprocessor", preprocessor()), ("classifier", XGBClassifier(
        scale_pos_weight=spw, n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="aucpr", n_jobs=-1))])


def lr():
    return Pipeline([("preprocessor", preprocessor()), ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42))])


def at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    n = len(y)
    return {"threshold": round(float(t), 6), "tp": tp, "fp": fp, "fn": fn,
            "recall": round(tp / max(tp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "per_10k": {"frauds_caught": round(tp / n * 1e4, 2), "frauds_missed": round(fn / n * 1e4, 2),
                        "false_alerts": round(fp / n * 1e4, 2)}}


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def table(y, p, extra):
    ts = sorted(set(THRESHOLDS + extra))
    return [at(y, p, t) for t in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "creditcard.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "threshold_analysis.json"))
    a = ap.parse_args()
    raw = Path(a.csv).read_bytes()
    df = pd.read_csv(a.csv)
    X, y = df.drop("Class", axis=1), df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    yv, yt = y_val.to_numpy(), y_te.to_numpy()

    out = {"csv_sha256": hashlib.sha256(raw).hexdigest(), "versions": {
        "python": sys.version.split()[0], "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "split": {"n_fit": len(y_fit), "n_val": len(y_val), "n_test": len(y_te),
                  "fraud_val": int(yv.sum()), "fraud_test": int(yt.sum())}, "models": {}}

    # 1) shipped artifact on the notebook's test split (headline check)
    shipped = joblib.load(ROOT / "artifacts" / "fraud_model_v1.joblib")
    ps = shipped.predict_proba(X_te)[:, 1]
    out["models"]["shipped_artifact"] = {
        "trained_on": "full train split (notebook)", "test_pr_auc": round(average_precision_score(yt, ps), 4),
        "test": table(yt, ps, [])}

    # 2) re-trained on fit split; threshold chosen on validation (max F1)
    for name, make in [("xgboost_fit_split", lambda: xgb((y_fit == 0).sum() / (y_fit == 1).sum())),
                       ("logreg_balanced_fit_split", lr)]:
        m = make().fit(X_fit, y_fit)
        pv, pt = m.predict_proba(X_val)[:, 1], m.predict_proba(X_te)[:, 1]
        t, f1 = best_f1_threshold(yv, pv)
        out["models"][name] = {
            "chosen_threshold_rule": "max F1 on validation", "chosen_threshold": round(t, 6),
            "val_f1_at_chosen": round(f1, 4), "val_pr_auc": round(average_precision_score(yv, pv), 4),
            "test_pr_auc": round(average_precision_score(yt, pt), 4),
            "test_at_chosen": at(yt, pt, t), "test": table(yt, pt, [])}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
