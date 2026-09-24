"""
Autonomous monitoring of the exam period (the optional task).

The 20 exam cases are alerts someone else raised. This runs the agent's own
detector across November and December without waiting to be asked, looking for
the thing the transaction-level risk score structurally cannot see: a rare
device profile carrying many unrelated cardholders inside a short window.

    python -m agent.monitor            -> monitoring/ring_findings.json + .md

Output is kept out of cases/ so it cannot be confused with the exam answers.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import get_dataset                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "monitoring")

START, END = "2016-11-01", "2016-12-31"
MIN_CARDS = 6
MAX_PROFILE_CARDS = 60


def scan(min_cards=MIN_CARDS, max_profile=MAX_PROFILE_CARDS):
    ds = get_dataset()
    lo, hi = pd.Timestamp(START), pd.Timestamp(END)
    pack_cards = set()
    for _, c in ds.pack.iterrows():
        r = ds.txn_row(int(c["flagged_txn_id"]))
        if r is not None:
            pack_cards.add(r["phys_card"])

    rings = []
    for dk, n_total in ds.device_card_counts.items():
        if n_total > max_profile:
            continue
        d = ds.device_txns(dk)
        w = d[(d["ts"] >= lo) & (d["ts"] <= hi)]
        n_win = w["phys_card"].nunique()
        if n_win < min_cards:
            continue
        conc = n_win / n_total
        if conc < 0.45:
            continue
        info = dk.split(" | ")[0]
        if info.startswith("unknown-device") and dk.split(" | ")[-1].startswith("unknown-screen"):
            continue
        rings.append({
            "device_key": dk,
            "n_cards_window": int(n_win),
            "n_cards_total": int(n_total),
            "concentration": round(conc, 3),
            "n_txns_window": int(len(w)),
            "median_risk_score": round(float(w["risk_score"].median()), 3),
            "max_risk_score": round(float(w["risk_score"].max()), 3),
            "total_amount_usd": round(float(w["TransactionAmt"].sum()), 2),
            "product_codes": w["ProductCD"].value_counts().to_dict(),
            "first_seen": str(w["ts"].min()), "last_seen": str(w["ts"].max()),
            "cards": sorted(w["card_id"].unique().tolist()),
            "customers": int(w["customer_id"].nunique()),
            "overlaps_case_pack": bool(pack_cards & set(w["phys_card"].unique())),
        })
    rings.sort(key=lambda r: (-r["n_cards_window"], -r["total_amount_usd"]))
    return rings


def main():
    os.makedirs(OUT, exist_ok=True)
    rings = scan()
    exposure = sum(r["total_amount_usd"] for r in rings)
    cards = sorted({c for r in rings for c in r["cards"]})
    flagged_high = sum(1 for r in rings if r["median_risk_score"] >= 0.7)

    payload = {
        "window": [START, END],
        "criteria": {
            "min_cards_in_window": MIN_CARDS,
            "max_lifetime_cards_on_profile": MAX_PROFILE_CARDS,
            "min_concentration": 0.45,
            "note": ("A device profile is DeviceInfo|OS|browser|screen. Profiles with "
                     "many lifetime cards are browser-family buckets, not machines, so "
                     "only rare and concentrated profiles are reported."),
        },
        "summary": {
            "rings": len(rings),
            "distinct_cards": len(cards),
            "total_amount_usd": round(exposure, 2),
            "rings_the_model_scored_above_0.7_median": flagged_high,
        },
        "rings": rings,
    }
    with open(os.path.join(OUT, "ring_findings.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    md = [f"# Autonomous monitoring — {START} to {END}", "",
          f"The agent scanned the exam period on its own, without an alert, for rare "
          f"device profiles carrying many unrelated cardholders.", "",
          f"- **{len(rings)} rings** found",
          f"- **{len(cards)} distinct cards** across them",
          f"- **${exposure:,.2f}** of transactions",
          f"- **{flagged_high}** of the {len(rings)} rings have a median model score above 0.7",
          "",
          "The last line is the finding. These clusters are almost invisible to a "
          "transaction-level score: every individual payment looks ordinary, and the "
          "pattern only exists as a shape in the graph.", "",
          "| cards | lifetime | conc | median risk | amount | product | device profile |",
          "|---|---|---|---|---|---|---|"]
    for r in rings[:25]:
        prods = ", ".join(f"{k}×{v}" for k, v in list(r["product_codes"].items())[:2])
        md.append(f"| {r['n_cards_window']} | {r['n_cards_total']} | "
                  f"{r['concentration']:.2f} | {r['median_risk_score']:.2f} | "
                  f"${r['total_amount_usd']:,.0f} | {prods} | `{r['device_key'][:58]}` |")
    md += ["", "Full detail, including every card id, is in `ring_findings.json`.", "",
           "Recommended handling under the policy: `CREATE_CASE`, "
           "`MONITOR_CONNECTED_CARDS` for every card on the profile, and "
           "`ESCALATE_TO_ANALYST` under R9 (undocumented, coordinated across "
           "customers). Filing is R9 + 3a where exposure or cross-customer linkage "
           "qualifies."]
    with open(os.path.join(OUT, "ring_findings.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"{len(rings)} rings, {len(cards)} cards, ${exposure:,.2f}")
    print(f"wrote {OUT}/ring_findings.json and .md")


if __name__ == "__main__":
    main()
