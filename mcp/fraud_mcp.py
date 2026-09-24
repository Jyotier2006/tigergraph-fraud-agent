"""
The investigation tools exposed over MCP.

Runs alongside the official TigerGraph MCP server (see tigergraph_mcp.json):
that one gives an agent raw graph access, this one gives it the *investigation*
verbs -- the traversals that already know what a fraud analyst is looking for,
with the rarity logic and the policy baked in.

    python -m mcp.fraud_mcp          # stdio MCP server

Tools
-----
  get_transaction        anchor an alert
  card_history           the cardholder's behavioural baseline
  card_window            what else happened on the card around the alert
  device_neighbors       shared-origin traversal, with rarity
  device_ring_scan       sweep a window for rare device profiles on many cards
  similar_prior_cases    case memory retrieval
  recurring_match        R7 check against the card's own repeat pattern
  policy_lookup          GraphRAG over the policy and typologies
  investigate_case       the whole loop, returning a finished case file
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp.server.fastmcp import FastMCP          # type: ignore
except Exception:                                    # pragma: no cover
    FastMCP = None

from agent.core import get_dataset                   # noqa: E402
from agent.graphrag import policy_index              # noqa: E402
from agent.tools import get_backend                  # noqa: E402

BACKEND = os.environ.get("FRAUD_BACKEND", "local")


def _be():
    return get_backend(BACKEND)


# --------------------------------------------------------------- tool bodies
def get_transaction(txn_id: str) -> dict:
    """Fetch one transaction with its device profile, region and match flags."""
    return _be().get_transaction(txn_id)


def card_history(card_id: str, before_ts: str = "") -> dict:
    """The card's behavioural baseline: amounts, products, channels, regions, devices."""
    import pandas as pd
    be = _be()
    return be.card_history(card_id, pd.Timestamp(before_ts) if before_ts else None)


def card_window(card_id: str, center_ts: str, hours: int = 72) -> dict:
    """Every transaction on the card within +/- hours of a point in time."""
    import pandas as pd
    return _be().card_window(card_id, pd.Timestamp(center_ts), hours)


def device_neighbors(device_key: str, center_ts: str, days: int = 15) -> dict:
    """
    Other cards on the same device profile, with the rarity numbers that decide
    whether the link is evidence: lifetime card count, window card count and
    concentration. A profile shared by hundreds of cards is a browser bucket.
    """
    import pandas as pd
    return _be().device_neighbors(device_key, pd.Timestamp(center_ts), days)


def device_ring_scan(start_ts: str, end_ts: str, min_cards: int = 3,
                     max_profile_cards: int = 60) -> dict:
    """
    Sweep a time window for rare device profiles carrying several unrelated
    cards -- the undocumented pattern. This is the query that finds rings the
    transaction-level risk score cannot see.
    """
    import pandas as pd
    ds = get_dataset()
    lo, hi = pd.Timestamp(start_ts), pd.Timestamp(end_ts)
    out = []
    for dk, n_total in ds.device_card_counts.items():
        if n_total > max_profile_cards:
            continue
        d = ds.device_txns(dk)
        w = d[(d["ts"] >= lo) & (d["ts"] <= hi)]
        n_win = w["phys_card"].nunique()
        if n_win < min_cards:
            continue
        out.append({"device_key": dk, "n_cards_window": int(n_win),
                    "n_cards_total": int(n_total),
                    "concentration": round(n_win / n_total, 3),
                    "n_txns_window": int(len(w)),
                    "median_risk": round(float(w["risk_score"].median()), 3),
                    "cards": sorted(w["card_id"].unique().tolist())[:40]})
    out.sort(key=lambda r: -r["n_cards_window"])
    return {"window": [start_ts, end_ts], "rings": out[:25]}


def similar_prior_cases(card_id: str, device_key: str = "", customer_id: str = "",
                        k: int = 6) -> dict:
    """Retrieve closed investigations linked to this alert by card, rare device or customer."""
    return _be().similar_prior_cases(card_id, [device_key] if device_key else [],
                                     customer_id, "", k)


def recurring_match(card_id: str, amount: float, product_cd: str,
                    before_ts: str) -> dict:
    """Policy R7: has this card already made this charge repeatedly?"""
    import pandas as pd
    return _be().recurring_match(card_id, amount, product_cd, pd.Timestamp(before_ts))


def policy_lookup(question: str, k: int = 3) -> dict:
    """GraphRAG over the fraud policy, the five typologies and the regulatory sections."""
    return {"chunks": policy_index().retrieve(question, k)}


def investigate_case(case_id: str) -> dict:
    """Run the full investigation loop for a case in the case pack."""
    from agent.investigate import investigate
    from agent.run_cases import finalize
    ds = get_dataset()
    row = ds.pack[ds.pack["case_id"] == case_id]
    if row.empty:
        return {"error": f"unknown case {case_id}"}
    ans = investigate(row.iloc[0], _be())
    ans["_internal"]["trigger_type"] = row.iloc[0]["trigger_type"]
    return finalize(ans)


TOOLS = [get_transaction, card_history, card_window, device_neighbors,
         device_ring_scan, similar_prior_cases, recurring_match, policy_lookup,
         investigate_case]


def main():                                            # pragma: no cover
    if FastMCP is None:
        print("mcp package not installed; `pip install mcp`", file=sys.stderr)
        # still usable as a library / CLI
        if len(sys.argv) > 2:
            fn = {f.__name__: f for f in TOOLS}[sys.argv[1]]
            print(json.dumps(fn(*sys.argv[2:]), indent=2, default=str))
        return
    server = FastMCP("fraud-investigation")
    for fn in TOOLS:
        server.tool()(fn)
    server.run()


if __name__ == "__main__":
    main()
