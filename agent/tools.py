"""
The agent's graph tools.

Every tool returns a JSON-serialisable dict and records a `ref` string of the
form ``query:name(args)`` so that each evidence item in the case file can point
back at the exact graph call that produced it.

Backends
--------
``TigerGraphBackend`` runs the installed GSQL queries over REST++ (production
path, see gsql/queries.gsql).  ``LocalBackend`` executes the identical
traversal semantics over the in-memory graph built by core.Dataset so the
investigation is reproducible without a live database.  Both satisfy the same
interface, and the agent does not know which one it is talking to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import get_dataset, M_COLS


def _f(v, nd=2):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return round(float(v), nd)
    except Exception:
        return None


def _txn_dict(r) -> dict:
    return {
        "txn_id": str(int(r["TransactionID"])),
        "ts": str(r["ts"]),
        "amount": _f(r["TransactionAmt"]),
        "product_cd": str(r["ProductCD"]),
        "channel": str(r["channel"]),
        "risk_score": _f(r["risk_score"]),
        "addr1": None if pd.isna(r["addr1"]) else float(r["addr1"]),
        "p_email": None if pd.isna(r["P_emaildomain"]) else str(r["P_emaildomain"]),
        "device_key": None if pd.isna(r.get("device_key")) else str(r.get("device_key")),
        "device_status": None if pd.isna(r.get("device_status")) else str(r.get("device_status")),
        "proxy_status": None if pd.isna(r.get("proxy_status")) else str(r.get("proxy_status")),
        "card_id": str(r["card_id"]),
        "phys_card": str(r["phys_card"]),
        "customer_id": str(r["customer_id"]),
    }


class LocalBackend:
    """Executes the graph queries over the in-memory graph."""

    name = "local"

    def __init__(self):
        self.ds = get_dataset()
        self.calls: list[str] = []

    # -- bookkeeping ------------------------------------------------------
    def _rec(self, ref: str) -> str:
        self.calls.append(ref)
        return ref

    @property
    def tool_calls(self) -> int:
        return len(self.calls)

    # -- Q1 ---------------------------------------------------------------
    def get_transaction(self, txn_id) -> dict:
        ref = self._rec(f"query:get_transaction(txn_id={txn_id})")
        r = self.ds.txn_row(txn_id)
        if r is None:
            return {"ref": ref, "found": False}
        d = _txn_dict(r)
        d["dist1"] = _f(r["dist1"])
        d["match_flags"] = {c: (None if pd.isna(r[c]) else str(r[c])) for c in M_COLS}
        d["ref"] = ref
        d["found"] = True
        return d

    # -- Q2 ---------------------------------------------------------------
    def card_history(self, phys_card: str, before_ts=None, limit=None) -> dict:
        ref = self._rec(f"query:card_history(card={phys_card})")
        h = self.ds.card_history(phys_card)
        if before_ts is not None:
            h = h[h["ts"] < before_ts]
        amts = h["TransactionAmt"]
        return {
            "ref": ref,
            "n_txns": int(len(h)),
            "first_ts": str(h["ts"].min()) if len(h) else None,
            "last_ts": str(h["ts"].max()) if len(h) else None,
            "amount_mean": _f(amts.mean()),
            "amount_median": _f(amts.median()),
            "amount_p95": _f(amts.quantile(0.95)) if len(h) else None,
            "amount_max": _f(amts.max()),
            "product_codes": h["ProductCD"].value_counts().to_dict(),
            "channels": h["channel"].value_counts().to_dict(),
            "regions": {str(k): int(v) for k, v in
                        h["addr1"].value_counts().head(12).to_dict().items()},
            "email_domains": {str(k): int(v) for k, v in
                              h["P_emaildomain"].value_counts().head(8).to_dict().items()},
            "device_keys": [str(x) for x in h["device_key"].dropna().unique()[:25]],
        }

    # -- Q3 ---------------------------------------------------------------
    def card_window(self, phys_card: str, center_ts, hours: float = 48) -> dict:
        ref = self._rec(f"query:card_window(card={phys_card}, hours={hours})")
        h = self.ds.card_history(phys_card)
        lo = center_ts - pd.Timedelta(hours=hours)
        hi = center_ts + pd.Timedelta(hours=hours)
        w = h[(h["ts"] >= lo) & (h["ts"] <= hi)].sort_values("ts")
        return {"ref": ref, "n": int(len(w)),
                "txns": [_txn_dict(r) for _, r in w.iterrows()]}

    # -- Q4 ---------------------------------------------------------------
    def device_neighbors(self, device_key: str, center_ts=None, days: int = 15) -> dict:
        """
        Other cards that used the same device profile (the shared-origin query).

        A device profile here is DeviceInfo + OS + browser + screen, so common
        combinations ("Windows | Windows 10 | chrome 63.0 | 1920x1080") are
        shared by hundreds of unrelated cardholders and mean nothing.  The query
        therefore returns both the lifetime card count and the count inside the
        window, plus whether the profile is *specific* (a concrete device model
        or a known screen).  A ring shows up as a specific, rare profile whose
        cards nearly all appear inside one short window.
        """
        ref = self._rec(f"query:device_neighbors(device_key={device_key!r}, days={days})")
        if not device_key or device_key == "nan" or not isinstance(device_key, str):
            return {"ref": ref, "found": False, "cards": []}
        all_d = self.ds.device_txns(device_key)
        d = all_d
        if center_ts is not None:
            lo = center_ts - pd.Timedelta(days=days)
            hi = center_ts + pd.Timedelta(days=days)
            d = all_d[(all_d["ts"] >= lo) & (all_d["ts"] <= hi)]
        cards = (d.groupby(["phys_card", "card_id", "customer_id"])
                  .agg(n=("TransactionID", "size"), first=("ts", "min"), last=("ts", "max"))
                  .reset_index())

        info, os_, br, scr = (device_key.split(" | ") + ["", "", "", ""])[:4]
        specific = (not info.startswith("unknown-device")) or (not scr.startswith("unknown-screen"))
        n_total = int(all_d["phys_card"].nunique())
        n_win = int(d["phys_card"].nunique())

        return {
            "ref": ref,
            "found": True,
            "device_key": device_key,
            "device_specific": bool(specific),
            "window_days": days,
            "n_txns_window": int(len(d)),
            "n_cards_window": n_win,
            "n_cards_total": n_total,
            "concentration": round(n_win / n_total, 3) if n_total else 0.0,
            "window_product_codes": d["ProductCD"].value_counts().to_dict(),
            "window_risk_median": _f(d["risk_score"].median()),
            "window_amount_median": _f(d["TransactionAmt"].median()),
            "cards": [{"phys_card": r.phys_card, "card_id": r.card_id,
                       "customer_id": r.customer_id, "n": int(r.n),
                       "first": str(r.first), "last": str(r.last)}
                      for r in cards.itertuples()],
        }

    # -- Q5 ---------------------------------------------------------------
    def region_cohort(self, addr1, center_ts, days: int = 14) -> dict:
        ref = self._rec(f"query:region_cohort(addr1={addr1}, days={days})")
        if addr1 is None or (isinstance(addr1, float) and np.isnan(addr1)):
            return {"ref": ref, "found": False}
        d = self.ds.region_txns(addr1)
        lo = center_ts - pd.Timedelta(days=days)
        hi = center_ts + pd.Timedelta(days=days)
        d = d[(d["ts"] >= lo) & (d["ts"] <= hi)]
        return {"ref": ref, "found": True, "addr1": float(addr1),
                "n_txns_window": int(len(d)),
                "n_cards_window": int(d["phys_card"].nunique())}

    # -- Q6 ---------------------------------------------------------------
    def similar_prior_cases(self, phys_card: str, device_keys: list[str],
                            customer_id: str, pattern_hint: str = "",
                            k: int = 6) -> dict:
        """Case memory: retrieve closed investigations connected to this alert."""
        ref = self._rec(f"query:similar_prior_cases(card={phys_card}, "
                        f"devices={len(device_keys)}, pattern={pattern_hint})")
        ds = self.ds
        scored: dict[str, tuple[float, list[str]]] = {}

        def bump(cid, w, why):
            s, ws = scored.get(cid, (0.0, []))
            scored[cid] = (s + w, ws + [why])

        for cid in ds.case_by_card.get(phys_card, []):
            bump(cid, 5.0, "same card")
        for dk in device_keys:
            # a device profile only identifies a machine when it is rare; the
            # common browser buckets link to hundreds of unrelated cases
            if ds.device_card_counts.get(dk, 0) > 60:
                continue
            for cid in ds.case_by_device.get(dk, []):
                bump(cid, 4.0, "same device profile")
        for _, r in ds.closed[ds.closed["customer_id"] == customer_id].iterrows():
            bump(r["case_id"], 3.0, "same customer")
        if pattern_hint:
            pool = ds.closed[(ds.closed["pattern"] == pattern_hint) &
                             (ds.closed["outcome"] == "confirmed_fraud")]
            for _, r in pool.head(400).iterrows():
                bump(r["case_id"], 0.8, f"same pattern ({pattern_hint})")

        top = sorted(scored.items(), key=lambda kv: -kv[1][0])[:k]
        out = []
        for cid, (sc, why) in top:
            r = ds.closed_by_id[cid]
            out.append({
                "case_id": cid, "score": round(sc, 2),
                "why": sorted(set(why)),
                "outcome": r["outcome"], "pattern": r["pattern"],
                "exposure_usd": _f(r["exposure_usd"]),
                "n_txns": int(r["n_txns"]) if not pd.isna(r["n_txns"]) else 0,
                "opened_at": str(r["opened_at"]), "closed_at": str(r["closed_at"]),
                "actions_taken": str(r["actions_taken"]),
                "report_filed": bool(r["report_filed"]) if not pd.isna(r["report_filed"]) else False,
                "analyst_notes": str(r["analyst_notes"])[:600],
            })
        return {"ref": ref, "cases": out}

    # -- Q7 ---------------------------------------------------------------
    def customer_cards(self, customer_id: str) -> dict:
        ref = self._rec(f"query:customer_cards(customer_id={customer_id})")
        h = self.ds.customer_history(customer_id)
        g = (h.groupby(["phys_card", "card_id"])
              .agg(n=("TransactionID", "size"), first=("ts", "min"), last=("ts", "max"))
              .reset_index())
        return {"ref": ref, "n_cards": int(len(g)),
                "cards": [{"phys_card": r.phys_card, "card_id": r.card_id, "n": int(r.n),
                           "first": str(r.first), "last": str(r.last)} for r in g.itertuples()]}

    # -- Q8 ---------------------------------------------------------------
    def recurring_match(self, phys_card: str, amount: float, product_cd: str,
                        center_ts) -> dict:
        """R7 support: does the disputed charge match the card's own recurring pattern?"""
        ref = self._rec(f"query:recurring_match(card={phys_card}, amount={amount})")
        h = self.ds.card_history(phys_card)
        prior = h[h["ts"] < center_ts]
        same = prior[(prior["ProductCD"] == product_cd) &
                     (np.abs(prior["TransactionAmt"] - amount) <= max(0.5, 0.02 * amount))]
        gaps = []
        if len(same) >= 2:
            t = same["ts"].sort_values()
            gaps = [round(g.total_seconds() / 86400.0, 1) for g in t.diff().dropna()]
        monthly = bool(gaps) and sum(25 <= g <= 35 for g in gaps) >= max(1, len(gaps) // 2)
        # The dataset carries no merchant field, so "same merchant, same amount"
        # (policy R7) is approximated by the same amount under the same product
        # code.  A charge the card has already made many times is an established
        # repeat whether or not the spacing is monthly.
        established = int(len(same)) >= 5
        span_days = 0.0
        if len(same) >= 2:
            span_days = round((same["ts"].max() - same["ts"].min()).total_seconds() / 86400.0, 1)
        return {"ref": ref, "n_prior_same_amount": int(len(same)),
                "gaps_days": gaps[-6:], "looks_monthly": monthly,
                "established_repeat": established, "span_days": span_days,
                "last_prior_ts": str(same["ts"].max()) if len(same) else None,
                "prior_txn_ids": [str(int(x)) for x in same["TransactionID"].tail(6)]}


def get_backend(kind: str = "local"):
    if kind == "tigergraph":
        from .tg_backend import TigerGraphBackend
        return TigerGraphBackend()
    return LocalBackend()
