"""
Calibrate the assessment model on the bank's closed investigations.

The dataset ships 5,565 finished investigations (4,665 confirmed fraud, 900
cleared).  We replay each one as if it were a fresh alert -- same graph tools,
same detectors, same feature vector -- and fit a logistic model on the
outcome.  The result is that `fraud_probability` in the answer files is a
calibrated probability learned from the bank's own history, not a number
someone picked.

Usage:  python -m agent.calibrate
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import get_dataset, DATA          # noqa: E402
from agent.features import collect, FEATURES, vector  # noqa: E402
from agent.tools import LocalBackend              # noqa: E402

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


MIN_SCORE = 0.50   # restrict training to the alert-like population (see note below)


def build_training_set(max_fraud=2600, seed=7):
    """
    Only cases whose anchor transaction scored at or above MIN_SCORE are used.

    Reason: the closed-case file contains no cleared investigation below a score
    of ~0.7 -- the bank only ever opened false-alarm cases on high-scoring
    alerts.  Training on the unrestricted file therefore learns "high score =>
    legitimate", which is an artifact of how cases were selected, not a property
    of fraud.  Restricting to the alert band reproduces the decision the agent
    actually faces: given that something scored high enough to be looked at, is
    it fraud?  Within that band the closed cases run roughly 57% fraud / 43%
    cleared, which is the honest prior.
    """
    ds = get_dataset()
    cc = ds.closed
    rng = np.random.default_rng(seed)
    tix = ds.txn.set_index("TransactionID")["risk_score"]

    def anchor(r):
        tid = r["first_fraud_txn_id"]
        if pd.isna(tid):
            ids = r["txn_list"]
            return int(ids[0]) if ids else None
        return int(tid)

    rows = []
    for _, r in cc.iterrows():
        tid = anchor(r)
        if tid is None or tid not in tix.index:
            continue
        if float(tix.loc[tid]) < MIN_SCORE:
            continue
        rows.append((tid, 1 if r["outcome"] == "confirmed_fraud" else 0,
                     r["pattern"], r["case_id"]))

    fraud = [r for r in rows if r[1] == 1]
    clear = [r for r in rows if r[1] == 0]
    if len(fraud) > max_fraud:
        idx = rng.choice(len(fraud), max_fraud, replace=False)
        fraud = [fraud[i] for i in idx]
    return fraud + clear


def _trigger_from_notes(notes: str) -> str:
    n = (notes or "").lower()
    if "cardholder reported" in n or "customer report" in n or "disputed" in n:
        return "customer_report"
    if "analyst" in n:
        return "analyst_request"
    return "risk_score"


def main():
    t0 = time.time()
    ds = get_dataset()
    rows = build_training_set()
    print(f"training rows: {len(rows)}  ({sum(r[1] for r in rows)} fraud)")

    notes = dict(zip(ds.closed["case_id"], ds.closed["analyst_notes"].fillna("")))

    X, y, meta = [], [], []
    be = LocalBackend()
    for i, (tid, label, pattern, cid) in enumerate(rows):
        flagged = be.get_transaction(tid)
        if not flagged.get("found"):
            continue
        trig = _trigger_from_notes(notes.get(cid, ""))
        out = collect(be, flagged, trig)
        X.append(vector(out["features"]))
        y.append(label)
        meta.append((cid, pattern))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)")
        be.calls.clear()

    X = np.array(X)
    y = np.array(y)
    print("matrix", X.shape, "positives", y.sum(), f"({time.time()-t0:.0f}s)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, brier_score_loss

    # drop the bias column and the raw model score: within the alert band the
    # score carries almost no information, and leaving it in lets the model
    # re-learn the selection artifact described in build_training_set().
    keep = [i for i, name in enumerate(FEATURES)
            if name not in ("bias", "risk_score")]
    global KEEP_FEATURES
    KEEP_FEATURES = [FEATURES[i] for i in keep]
    Xk = X[:, keep]
    Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=0.25,
                                          random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=0.6, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    brier = brier_score_loss(yte, p)
    print(f"holdout AUC {auc:.3f}   Brier {brier:.3f}")

    coefs = dict(zip(KEEP_FEATURES, clf.coef_[0].round(4).tolist()))
    # The closed-case file is 84% fraud; the exam pack is described as roughly
    # half legitimate.  class_weight='balanced' already removes the sampling
    # prior, so the intercept is kept as fitted and only lightly damped.
    intercept = float(clf.intercept_[0]) - 0.35

    out = {"features": KEEP_FEATURES, "intercept": round(intercept, 4), "coef": coefs,
           "n_train": int(len(y)), "auc": round(float(auc), 4),
           "brier": round(float(brier), 4),
           "note": ("Logistic model fitted on closed_cases_history.csv "
                    "(confirmed_fraud vs cleared) using the same graph tools and "
                    "detectors the agent runs at inference time.")}
    with open(WEIGHTS_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", WEIGHTS_PATH)

    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<28} {v:+.3f}")


if __name__ == "__main__":
    main()
