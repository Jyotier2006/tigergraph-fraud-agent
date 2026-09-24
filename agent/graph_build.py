"""
Emit the vertex / edge CSVs that gsql/02_load.gsql loads into TigerGraph.

    python -m agent.graph_build                # full dataset
    python -m agent.graph_build --subgraph     # investigation neighbourhood only

The subgraph mode keeps every entity the 20 exam cases actually touch (their
cards' full histories, every card that shares a ring device profile, and all
5,565 closed cases) and is what the demo workspace is seeded with; the full
mode is the production load.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.core import get_dataset, DATA, M_COLS   # noqa: E402

OUT = os.path.join(DATA, "graph")
RARE_PROFILE_CARDS = 60


def _mkdir():
    os.makedirs(OUT, exist_ok=True)


def build(subgraph: bool = False, window_days: int = 120):
    ds = get_dataset()
    _mkdir()
    txn = ds.txn

    if subgraph:
        keep_cards: set[str] = set()
        keep_rows: list[np.ndarray] = []
        for _, c in ds.pack.iterrows():
            r = ds.txn_row(int(c["flagged_txn_id"]))
            if r is None:
                continue
            keep_cards.add(r["phys_card"])
            # every card that shares a rare device profile with this alert
            dk = r.get("device_key")
            if isinstance(dk, str) and ds.device_card_counts.get(dk, 0) <= RARE_PROFILE_CARDS:
                d = ds.device_txns(dk)
                keep_cards.update(d["phys_card"].unique())
        # all cards named by a closed case (so memory retrieval works)
        keep_cards.update(ds.case_by_card.keys())
        for pc in keep_cards:
            idx = ds.card_rows.get(pc)
            if idx is not None:
                keep_rows.append(idx)
        rows = np.unique(np.concatenate(keep_rows)) if keep_rows else np.array([], dtype=int)
        txn = txn.iloc[rows]
        print(f"subgraph: {len(keep_cards):,} cards, {len(txn):,} transactions")

    # ---------------------------------------------------------- Transaction
    tx = pd.DataFrame({
        "txn_id": txn["TransactionID"].astype(str),
        "ts": txn["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": txn["TransactionAmt"].round(2),
        "product_cd": txn["ProductCD"].fillna(""),
        "channel": txn["channel"].fillna(""),
        "risk_score": txn["risk_score"].fillna(0).round(4),
        "addr1": txn["addr1"].apply(lambda v: "" if pd.isna(v) else str(v)),
        "addr2": txn["addr2"].apply(lambda v: "" if pd.isna(v) else str(v)),
        "dist1": txn["dist1"].fillna(-1),
        "p_email": txn["P_emaildomain"].fillna(""),
        "r_email": txn["R_emaildomain"].fillna(""),
        "device_status": txn.get("device_status", pd.Series(index=txn.index)).fillna(""),
        "proxy_status": txn.get("proxy_status", pd.Series(index=txn.index)).fillna(""),
        "match_false_count": (txn[M_COLS] == "F").sum(axis=1),
        "customer_id": txn["customer_id"],
        "card_id": txn["card_id"],
    })
    tx.to_csv(f"{OUT}/transaction.csv", index=False)

    # ----------------------------------------------------------------- Card
    g = txn.groupby(["phys_card", "card_id", "customer_id"])
    card = g.agg(n_txns=("TransactionID", "size"),
                 first_seen=("ts", "min"), last_seen=("ts", "max"),
                 amount_median=("TransactionAmt", "median"),
                 amount_p95=("TransactionAmt", lambda s: s.quantile(0.95))).reset_index()
    meta = txn.groupby("phys_card").agg(card1=("card1", "first"), network=("card4", "first"),
                                        card_type=("card6", "first")).reset_index()
    card = card.merge(meta, on="phys_card", how="left")
    card["card1"] = card["card1"].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    card["network"] = card["network"].fillna("")
    card["card_type"] = card["card_type"].fillna("")
    card["first_seen"] = card["first_seen"].dt.strftime("%Y-%m-%d %H:%M:%S")
    card["last_seen"] = card["last_seen"].dt.strftime("%Y-%m-%d %H:%M:%S")
    card[["card_id", "phys_card", "customer_id", "card1", "network", "card_type",
          "n_txns", "first_seen", "last_seen", "amount_median", "amount_p95"]] \
        .round({"amount_median": 2, "amount_p95": 2}).to_csv(f"{OUT}/card.csv", index=False)

    # ------------------------------------------------------------- Customer
    cu = txn.groupby("customer_id").agg(n_txns=("TransactionID", "size"),
                                        first_seen=("ts", "min"), last_seen=("ts", "max"),
                                        total_spend=("TransactionAmt", "sum")).reset_index()
    cu["n_cards"] = txn.groupby("customer_id")["phys_card"].nunique().values
    home = (txn.dropna(subset=["addr1"]).groupby("customer_id")["addr1"]
              .agg(lambda s: s.value_counts().idxmax() if len(s) else ""))
    cu["home_region"] = cu["customer_id"].map(home).apply(lambda v: "" if pd.isna(v) else str(v))
    cu["first_seen"] = cu["first_seen"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cu["last_seen"] = cu["last_seen"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cu[["customer_id", "n_cards", "n_txns", "first_seen", "last_seen", "home_region",
        "total_spend"]].round({"total_spend": 2}).to_csv(f"{OUT}/customer.csv", index=False)

    # -------------------------------------------------------- DeviceProfile
    dev = txn.dropna(subset=["device_key"])
    if len(dev):
        dp = dev.groupby("device_key").agg(n_txns=("TransactionID", "size"),
                                           device_type=("DeviceType", "first")).reset_index()
        # lifetime card count comes from the FULL dataset even in subgraph mode:
        # rarity is a property of the profile, not of the slice we loaded.
        dp["n_cards_total"] = dp["device_key"].map(ds.device_card_counts).fillna(0).astype(int)
        parts = dp["device_key"].str.split(" | ", regex=False)
        dp["device_info"] = parts.str[0]
        dp["os"] = parts.str[1]
        dp["browser"] = parts.str[2]
        dp["screen"] = parts.str[3]
        dp["is_specific"] = ((~dp["device_info"].str.startswith("unknown-device")) |
                             (~dp["screen"].fillna("").str.startswith("unknown-screen")))
        dp["device_type"] = dp["device_type"].fillna("")
        dp[["device_key", "device_info", "os", "browser", "screen", "device_type",
            "n_txns", "n_cards_total", "is_specific"]].to_csv(f"{OUT}/device_profile.csv", index=False)

    # -------------------------------------------- EmailDomain / BillingRegion
    em = txn["P_emaildomain"].dropna().value_counts().reset_index()
    em.columns = ["domain", "n_txns"]
    em.to_csv(f"{OUT}/email_domain.csv", index=False)

    rg = txn.dropna(subset=["addr1"]).groupby("addr1").agg(
        n_txns=("TransactionID", "size"), n_cards=("phys_card", "nunique"),
        country=("addr2", "first")).reset_index()
    rg["region_code"] = rg["addr1"].astype(str)
    rg["country"] = rg["country"].apply(lambda v: "" if pd.isna(v) else str(v))
    rg[["region_code", "country", "n_txns", "n_cards"]].to_csv(f"{OUT}/billing_region.csv", index=False)

    # ------------------------------------------------------------ edges ----
    pd.DataFrame({"customer_id": card["customer_id"], "card_id": card["card_id"]}) \
        .to_csv(f"{OUT}/e_owns.csv", index=False)
    pd.DataFrame({"card_id": txn["card_id"], "txn_id": txn["TransactionID"].astype(str)}) \
        .to_csv(f"{OUT}/e_made.csv", index=False)
    if len(dev):
        pd.DataFrame({"txn_id": dev["TransactionID"].astype(str),
                      "device_key": dev["device_key"]}).to_csv(f"{OUT}/e_from_device.csv", index=False)
    e = txn.dropna(subset=["P_emaildomain"])
    pd.DataFrame({"txn_id": e["TransactionID"].astype(str),
                  "domain": e["P_emaildomain"]}).to_csv(f"{OUT}/e_purchaser_email.csv", index=False)
    r = txn.dropna(subset=["addr1"])
    pd.DataFrame({"txn_id": r["TransactionID"].astype(str),
                  "region_code": r["addr1"].astype(str)}).to_csv(f"{OUT}/e_billed_in.csv", index=False)

    # NEXT: consecutive transactions on a card, ordered by ts
    s = txn.sort_values(["phys_card", "ts"])
    ids = s["TransactionID"].astype("int64")
    nxt = pd.DataFrame({"a": ids.astype(str),
                        "b": ids.shift(-1).fillna(-1).astype("int64").astype(str),
                        "same": s["phys_card"].eq(s["phys_card"].shift(-1))})
    nxt[nxt["same"]][["a", "b"]].to_csv(f"{OUT}/e_next.csv", index=False)

    # -------------------------------------------------------- ClosedCase ---
    cc = ds.closed
    out_cc = pd.DataFrame({
        "case_id": cc["case_id"], "customer_id": cc["customer_id"], "card_id": cc["card_id"],
        "opened_at": cc["opened_at"], "closed_at": cc["closed_at"], "outcome": cc["outcome"],
        "pattern": cc["pattern"], "n_txns": cc["n_txns"].fillna(0).astype(int),
        "exposure_usd": cc["exposure_usd"].fillna(0).round(2),
        "actions_taken": cc["actions_taken"].fillna(""),
        "report_filed": cc["report_filed"].map(
            lambda v: 1 if str(v).strip().lower() in ("yes", "true", "1") else 0),
        "analyst_notes": cc["analyst_notes"].fillna("").str.replace("\n", " ", regex=False),
    })
    out_cc.to_csv(f"{OUT}/closed_case.csv", index=False)

    known = set(tx["txn_id"])
    inv = [(r["case_id"], str(t)) for _, r in cc.iterrows() for t in r["txn_list"]
           if str(t) in known]
    pd.DataFrame(inv, columns=["case_id", "txn_id"]).to_csv(f"{OUT}/e_involves.csv", index=False)
    pd.DataFrame({"case_id": cc["case_id"], "card_id": cc["card_id"]}) \
        .to_csv(f"{OUT}/e_on_card.csv", index=False)
    conn = []
    for _, r in cc.iterrows():
        for c in str(r["connected_card_ids"] or "").split("|"):
            if c.strip():
                conn.append((r["case_id"], c.strip()))
    pd.DataFrame(conn, columns=["case_id", "card_id"]).to_csv(f"{OUT}/e_connected_to.csv", index=False)

    print(f"wrote graph CSVs to {OUT}")
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"  {f:<28} {os.path.getsize(p)/1e6:8.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", action="store_true")
    a = ap.parse_args()
    build(subgraph=a.subgraph)
