"""
Core data layer for the Agentic Fraud Investigation system.

Loads the HHGOA / IEEE-CIS dataset, reconstructs the entities that the graph
schema is built on (Customer, Card, Transaction, DeviceProfile, EmailDomain,
BillingRegion, ClosedCase) and exposes the indices the investigation tools
traverse.

The same entity definitions are used by:
  * graph_build.py   -> emits the vertex/edge CSVs loaded into TigerGraph
  * tools.py         -> the agent's graph tools (TigerGraph or local backend)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from functools import cached_property

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")

# Readable identity columns (per the dataset README).
ID_READABLE = {
    "id_12": "id_12",
    "id_15": "device_status",     # New / Found
    "id_23": "proxy_status",      # transparent / anonymous / hidden
    "id_30": "os",
    "id_31": "browser",
    "id_33": "screen",
    "id_34": "match_status",
}

M_COLS = [f"M{i}" for i in range(1, 10)]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]


def _s(v):
    """Normalise a value to a clean string ('' for NaN)."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


@dataclass
class Dataset:
    """In-memory graph-shaped view of the dataset."""

    txn: pd.DataFrame = field(repr=False, default=None)
    ident: pd.DataFrame = field(repr=False, default=None)
    closed: pd.DataFrame = field(repr=False, default=None)
    pack: pd.DataFrame = field(repr=False, default=None)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls) -> "Dataset":
        txn = pd.read_parquet(os.path.join(DATA, "txn.parquet"))
        ident = pd.read_parquet(os.path.join(DATA, "ident.parquet"))
        closed = pd.read_csv(os.path.join(RAW, "closed_cases_history.csv"))
        pack = pd.read_csv(os.path.join(RAW, "case_pack.csv"))

        txn["ts"] = pd.to_datetime(txn["ts"])
        txn["TransactionID"] = txn["TransactionID"].astype(np.int64)
        ident["TransactionID"] = ident["TransactionID"].astype(np.int64)

        ds = cls(txn=txn, ident=ident, closed=closed, pack=pack)
        ds._build()
        return ds

    # ----------------------------------------------------------------- build
    def _build(self):
        txn, ident = self.txn, self.ident

        # ---- device profile key: DeviceInfo | OS | browser | screen --------
        di = ident.rename(columns=ID_READABLE)
        for c in ["DeviceInfo", "os", "browser", "screen", "device_status",
                  "proxy_status", "match_status", "DeviceType"]:
            if c not in di.columns:
                di[c] = np.nan
        di["device_key"] = (
            di["DeviceInfo"].fillna("unknown-device").astype(str) + " | "
            + di["os"].fillna("unknown-os").astype(str) + " | "
            + di["browser"].fillna("unknown-browser").astype(str) + " | "
            + di["screen"].fillna("unknown-screen").astype(str)
        )
        self.ident_n = di

        # ---- attach identity to transactions --------------------------------
        cols = ["TransactionID", "device_key", "DeviceType", "DeviceInfo", "os",
                "browser", "screen", "device_status", "proxy_status",
                "match_status", "id_01", "id_02", "id_05", "id_06"]
        cols = [c for c in cols if c in di.columns]
        txn = txn.merge(di[cols], on="TransactionID", how="left")

        # ---- physical card = (customer_id, card1) ---------------------------
        # The K-suffix in the published card_ids (e.g. C01234-K1) is an issuer
        # label that is NOT recoverable from the transaction columns: customers
        # carry K1 and K2 labels over an identical card1. We therefore key the
        # physical card on (customer_id, card1) -- which is 100% pure against
        # the closed-case transaction sets -- and carry the published label as
        # an alias on the Card vertex.
        txn["card1_s"] = txn["card1"].astype("Int64").astype(str)
        txn["phys_card"] = txn["customer_id"].astype(str) + "-" + txn["card1_s"]

        order = (txn.groupby(["customer_id", "phys_card"], sort=False)["TransactionDT"]
                    .min().reset_index().sort_values(["customer_id", "TransactionDT"]))
        order["k"] = order.groupby("customer_id").cumcount() + 1
        kmap = dict(zip(order["phys_card"], order["k"]))
        txn["card_id"] = [f"{c}-K{kmap[p]}" for c, p in
                          zip(txn["customer_id"], txn["phys_card"])]

        txn = txn.sort_values("ts").reset_index(drop=True)
        self.txn = txn

        # ---- indices --------------------------------------------------------
        # NOTE: use .groups (index labels), not .indices (positions *within the
        # grouped frame*).  txn carries a RangeIndex after reset_index, so the
        # labels are valid positions for .iloc on the full frame even when the
        # group was built from a filtered view.
        self.by_tid = dict(zip(txn["TransactionID"], txn.index))

        def _groups(frame, key):
            return {k: np.asarray(v) for k, v in frame.groupby(key).groups.items()}

        self.card_rows = _groups(txn, "phys_card")
        self.cust_rows = _groups(txn, "customer_id")
        self.device_rows = _groups(txn.dropna(subset=["device_key"]), "device_key")
        self.region_rows = _groups(txn.dropna(subset=["addr1"]), "addr1")

        # ---- closed-case memory --------------------------------------------
        cc = self.closed.copy()
        cc["txn_list"] = cc["txn_ids"].fillna("").astype(str).apply(
            lambda s: [int(x) for x in s.split("|") if x.strip().isdigit()])
        cc["opened_dt"] = pd.to_datetime(cc["opened_at"], errors="coerce")
        self.closed = cc

        # closed case -> the physical cards / devices it touched
        tid2row = self.by_tid
        cc_cards, cc_devs = [], []
        for ids in cc["txn_list"]:
            rows = [tid2row[i] for i in ids if i in tid2row]
            cc_cards.append(sorted(set(txn["phys_card"].iloc[rows])) if rows else [])
            d = [x for x in txn["device_key"].iloc[rows].dropna().unique()] if rows else []
            cc_devs.append(sorted(set(d)))
        cc["phys_cards"] = cc_cards
        cc["device_keys"] = cc_devs

        # reverse indices for memory retrieval
        self.case_by_card: dict[str, list[str]] = {}
        self.case_by_device: dict[str, list[str]] = {}
        for _, r in cc.iterrows():
            for pc in r["phys_cards"]:
                self.case_by_card.setdefault(pc, []).append(r["case_id"])
            for dk in r["device_keys"]:
                self.case_by_device.setdefault(dk, []).append(r["case_id"])
        self.closed_by_id = {r["case_id"]: r for _, r in cc.iterrows()}

    # ------------------------------------------------------------- accessors
    def txn_row(self, tid: int):
        idx = self.by_tid.get(int(tid))
        return None if idx is None else self.txn.iloc[idx]

    def card_history(self, phys_card: str) -> pd.DataFrame:
        idx = self.card_rows.get(phys_card)
        return self.txn.iloc[idx] if idx is not None else self.txn.iloc[[]]

    def customer_history(self, customer_id: str) -> pd.DataFrame:
        idx = self.cust_rows.get(customer_id)
        return self.txn.iloc[idx] if idx is not None else self.txn.iloc[[]]

    def device_txns(self, device_key: str) -> pd.DataFrame:
        idx = self.device_rows.get(device_key)
        return self.txn.iloc[idx] if idx is not None else self.txn.iloc[[]]

    def region_txns(self, addr1) -> pd.DataFrame:
        idx = self.region_rows.get(addr1)
        return self.txn.iloc[idx] if idx is not None else self.txn.iloc[[]]

    @cached_property
    def device_card_counts(self) -> dict:
        """device_key -> number of distinct physical cards seen on it."""
        d = self.txn.dropna(subset=["device_key"])
        return d.groupby("device_key")["phys_card"].nunique().to_dict()


_DS: Dataset | None = None


def get_dataset() -> Dataset:
    global _DS
    if _DS is None:
        _DS = Dataset.load()
    return _DS
