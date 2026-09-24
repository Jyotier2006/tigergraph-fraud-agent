"""
Feature extraction shared by calibration and inference.

`collect` runs the graph tools for one alert and returns both the human-readable
signals (which become case evidence) and the numeric vector the calibrated
model scores.  Using one function for both guarantees the weights learned on
the bank's closed cases are the weights applied to the exam cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import detectors as D

FEATURES = [
    "bias",
    "trigger_customer_report",
    "trigger_analyst",
    "card_testing",
    "out_of_region_home_cont",
    "out_of_region_trip",
    "device_new_to_card",
    "device_status_new",
    "proxy_hidden",
    "device_shared_cards",
    "device_ring",
    "device_concentration",
    "amount_ratio_log",
    "amount_over_max",
    "product_novel",
    "channel_novel",
    "email_novel",
    "burst_odd",
    "match_flags_false",
    "prior_fraud_link",
    "prior_cleared_link",
    "risk_score",
    "hist_thin",
    "night_hour",
    "dist_novel",
    "recurring_match",
]


def collect(be, flagged, trigger: str, official_card_id: str = "") -> dict:
    """Run the graph traversals for one alert and build signals + features."""
    phys = flagged["phys_card"]
    ts = pd.Timestamp(flagged["ts"])

    hist = be.card_history(phys, before_ts=ts)
    win = be.card_window(phys, ts, hours=72)
    dev = be.device_neighbors(flagged.get("device_key") or "", ts, days=30)
    mem = be.similar_prior_cases(
        phys, [flagged["device_key"]] if flagged.get("device_key") else [],
        flagged["customer_id"])
    rec = be.recurring_match(phys, flagged["amount"], flagged["product_cd"], ts)

    sigs = {
        "card_testing": D.card_testing(win, flagged, hist),
        "out_of_region": D.out_of_region(flagged, hist, win),
        "new_device": D.new_device(flagged, hist, dev),
        "shared_device": D.shared_device(dev, phys),
        "device_ring": D.device_ring(dev, phys),
        "amount_anomaly": D.amount_anomaly(flagged, hist),
        "product_novelty": D.product_novelty(flagged, hist),
        "channel_novelty": D.channel_novelty(flagged, hist),
        "email_novelty": D.email_novelty(flagged, hist),
        "burst": D.burst(win, flagged, hist),
        "match_flag_anomaly": D.match_flag_anomaly(flagged),
        "prior_cases": D.prior_case_signal(mem, phys),
        "risk_score": D.risk_score_signal(flagged),
        "recurring_charge": D.recurring_charge(rec, flagged),
    }

    oor = sigs["out_of_region"]
    nd = sigs["new_device"]
    pc = sigs["prior_cases"]
    amt = sigs["amount_anomaly"]
    hist_n = hist.get("n_txns", 0)
    med = hist.get("amount_median") or 1.0
    ratio = (flagged.get("amount") or 0) / med if med else 1.0
    hour = ts.hour
    known_dists = None

    f = {
        "bias": 1.0,
        "trigger_customer_report": 1.0 if trigger == "customer_report" else 0.0,
        "trigger_analyst": 1.0 if trigger == "analyst_request" else 0.0,
        "card_testing": 1.0 if sigs["card_testing"].fired else 0.0,
        "out_of_region_home_cont": 1.0 if (oor.fired and
                                           oor.detail.get("home_activity_during", 0) > 0) else 0.0,
        "out_of_region_trip": 1.0 if (oor.fired and
                                      oor.detail.get("home_activity_during", 0) == 0) else 0.0,
        "device_new_to_card": 1.0 if (flagged.get("device_key") and
                                      flagged["device_key"] not in set(hist.get("device_keys", []))) else 0.0,
        "device_status_new": 1.0 if (flagged.get("device_status") == "New") else 0.0,
        "proxy_hidden": 1.0 if str(flagged.get("proxy_status") or "").lower() in
                               ("anonymous", "hidden") else 0.0,
        "device_shared_cards": float(min(len([c for c in dev.get("cards", [])
                                              if c["phys_card"] != phys]), 5))
                               if dev.get("device_specific") else 0.0,
        "device_ring": 1.0 if sigs["device_ring"].fired else 0.0,
        "device_concentration": float(dev.get("concentration", 0.0) or 0.0)
                                if dev.get("device_specific") else 0.0,
        "amount_ratio_log": float(np.clip(np.log1p(max(ratio, 0.01)), 0, 6)),
        "amount_over_max": 1.0 if (flagged.get("amount") or 0) > (hist.get("amount_max") or 1e9) else 0.0,
        "product_novel": 1.0 if sigs["product_novelty"].fired else 0.0,
        "channel_novel": 1.0 if sigs["channel_novelty"].fired else 0.0,
        "email_novel": 1.0 if sigs["email_novelty"].fired else 0.0,
        "burst_odd": 1.0 if sigs["burst"].fired else 0.0,
        "match_flags_false": float(min(len(flagged.get("match_flags", {}) and
                                           [k for k, v in (flagged.get("match_flags") or {}).items()
                                            if v == "F"]), 4)),
        "prior_fraud_link": 1.0 if (pc.fired and pc.weight > 0) else 0.0,
        "prior_cleared_link": 1.0 if (pc.fired and pc.weight < 0) else 0.0,
        "risk_score": float(flagged.get("risk_score") or 0.0),
        "hist_thin": 1.0 if hist_n < 10 else 0.0,
        "night_hour": 1.0 if (hour <= 5) else 0.0,
        "dist_novel": 0.0,
        "recurring_match": 1.0 if sigs["recurring_charge"].fired else 0.0,
    }
    return {"features": f, "signals": sigs, "hist": hist, "win": win, "dev": dev,
            "mem": mem, "rec": rec}


def vector(f: dict) -> np.ndarray:
    return np.array([f[k] for k in FEATURES], dtype=float)
