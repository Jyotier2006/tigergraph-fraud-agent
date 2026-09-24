"""
TigerGraph-backed implementation of the agent's tools.

Same interface as tools.LocalBackend: every method runs the matching installed
GSQL query (gsql/03_queries.gsql) and returns the same dict shape, so the
investigation logic is identical whichever backend is in use.  That is what
makes `--backend tigergraph` a drop-in switch rather than a second codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tg_client import from_env
from .tools import LocalBackend


class TigerGraphBackend(LocalBackend):
    """Runs the installed queries; falls back to the local graph per-call."""

    name = "tigergraph"

    def __init__(self, client=None, strict: bool = False):
        super().__init__()
        self.client = client or from_env()
        self.strict = strict

    def _q(self, name: str, params: dict):
        return self.client.run_query(name, params)

    # ------------------------------------------------------------------
    def get_transaction(self, txn_id) -> dict:
        ref = self._rec(f"query:get_transaction(txn_id={txn_id})")
        try:
            res = self._q("get_transaction", {"txn": str(txn_id)})
            t = res[0]["transaction"][0]["attributes"]
            dev = (res[1].get("device_profile") or [None])[0]
            return {
                "ref": ref, "found": True,
                "txn_id": str(txn_id), "ts": t["ts"], "amount": t["amount"],
                "product_cd": t["product_cd"], "channel": t["channel"],
                "risk_score": t["risk_score"],
                "addr1": float(t["addr1"]) if t.get("addr1") else None,
                "p_email": t.get("p_email") or None,
                "device_key": dev, "device_status": t.get("device_status") or None,
                "proxy_status": t.get("proxy_status") or None,
                "card_id": t["card_id"], "customer_id": t["customer_id"],
                "phys_card": t["card_id"],
                "match_flags": {"count_false": t.get("match_false_count", 0)},
            }
        except Exception:
            if self.strict:
                raise
            return super().get_transaction(txn_id)

    def card_history(self, phys_card: str, before_ts=None, limit=None) -> dict:
        ref = self._rec(f"query:card_history(card={phys_card})")
        try:
            res = self._q("card_history", {"card": phys_card,
                                           "before_ts": str(before_ts)})
            g = {k: v for d in res for k, v in d.items()}
            return {"ref": ref, "n_txns": g.get("n_txns", 0),
                    "amount_max": g.get("amount_max"),
                    "product_codes": {p: 1 for p in g.get("product_codes", [])},
                    "channels": {c: 1 for c in g.get("channels", [])},
                    "regions": {r: 1 for r in g.get("regions", []) if r},
                    "email_domains": {e: 1 for e in g.get("email_domains", []) if e},
                    "device_keys": g.get("device_keys", []),
                    "amount_mean": None, "amount_median": None, "amount_p95": None,
                    "first_ts": None, "last_ts": None}
        except Exception:
            if self.strict:
                raise
            return super().card_history(phys_card, before_ts, limit)

    def device_neighbors(self, device_key: str, center_ts=None, days: int = 15) -> dict:
        ref = self._rec(f"query:device_neighbors(device_key={device_key!r}, days={days})")
        try:
            res = self._q("device_neighbors", {"device": device_key,
                                               "center_ts": str(center_ts), "days": days})
            g = {k: v for d in res for k, v in d.items()}
            dev = (g.get("device") or [{}])[0].get("attributes", {})
            cards = g.get("cards_in_window", [])
            return {"ref": ref, "found": True, "device_key": device_key,
                    "device_specific": bool(dev.get("is_specific")),
                    "window_days": days,
                    "n_cards_window": g.get("n_cards_window", len(cards)),
                    "n_cards_total": dev.get("n_cards_total", 0),
                    "n_txns_window": g.get("n_txns_window", 0),
                    "concentration": round(len(cards) / dev["n_cards_total"], 3)
                                     if dev.get("n_cards_total") else 0.0,
                    "window_risk_median": g.get("median_risk_window"),
                    "window_product_codes": {},
                    "cards": [{"phys_card": c, "card_id": c, "customer_id": c.split("-")[0],
                               "n": 1, "first": "", "last": ""} for c in cards]}
        except Exception:
            if self.strict:
                raise
            return super().device_neighbors(device_key, center_ts, days)

    def similar_prior_cases(self, phys_card, device_keys, customer_id,
                            pattern_hint="", k: int = 6) -> dict:
        ref = self._rec(f"query:similar_prior_cases(card={phys_card}, k={k})")
        try:
            res = self._q("similar_prior_cases", {"card": phys_card, "k": k})
            rows = (res[0].get("prior_cases") if res else []) or []
            out = []
            for r in rows:
                a = r["attributes"]
                out.append({"case_id": r["v_id"], "score": a.get("@score", 0),
                            "why": a.get("@why", []), "outcome": a.get("outcome"),
                            "pattern": a.get("pattern"),
                            "exposure_usd": a.get("exposure_usd"),
                            "n_txns": a.get("n_txns", 0),
                            "opened_at": a.get("opened_at"), "closed_at": a.get("closed_at"),
                            "actions_taken": a.get("actions_taken", ""),
                            "report_filed": bool(a.get("report_filed")),
                            "analyst_notes": (a.get("analyst_notes") or "")[:600]})
            return {"ref": ref, "cases": out}
        except Exception:
            if self.strict:
                raise
            return super().similar_prior_cases(phys_card, device_keys, customer_id,
                                               pattern_hint, k)
