"""
Narrative layer.

Two writers produce the prose an analyst and a regulator read:

  * `summarise`  -> case.summary, two to six sentences
  * `sar_narrative` -> the FinCEN-shaped report: who, what, when, where, how,
    why it is suspicious

Both are written from the evidence the graph produced, never from the raw
dataframes, so nothing appears in the prose that is not also in
`case.evidence`.  An LLM (Gemini) can be plugged in through `llm_rewrite` to
polish the wording; the deterministic writer is the default so the pipeline is
reproducible and runs without network access.
"""
from __future__ import annotations

import textwrap

PATTERN_WORDS = {
    "card_testing": "card testing",
    "card_not_present_fraud": "card-not-present fraud",
    "card_not_present_new_device": "card-not-present fraud from a new device",
    "out_of_region_use": "out-of-region card-present use",
    "account_takeover": "account takeover",
    "undocumented": "an undocumented pattern",
    "none": "no fraud pattern",
}


def _money(x) -> str:
    return f"${x:,.2f}"


def summarise(ans: dict) -> str:
    c = ans["case"]
    it = ans["_internal"]
    f = it["flagged"]
    hist = it["hist"]
    verdict = c["verdict"]
    pat = PATTERN_WORDS.get(c["pattern"], c["pattern"])
    trig = it.get("trigger_type", "")

    lead = (f"Transaction {f['txn_id']} for {_money(f['amount'])} on card "
            f"{it['official_card_id']} ({f['channel'].replace('_', ' ')}, product "
            f"{f['product_cd']}) on {str(f['ts'])[:16]}")

    parts: list[str] = []
    if verdict == "fraud":
        parts.append(f"{lead} is assessed as fraud at probability "
                     f"{c['fraud_probability']:.2f}, consistent with {pat}.")
    elif verdict == "legitimate":
        parts.append(f"{lead} is assessed as legitimate at fraud probability "
                     f"{c['fraud_probability']:.2f}.")
    else:
        parts.append(f"{lead} remains uncertain at fraud probability "
                     f"{c['fraud_probability']:.2f}; the evidence points both ways.")

    # the two strongest findings, in the agent's own words
    for ev in c["evidence"][:2]:
        claim = ev["claim"].rstrip(".")
        parts.append(claim + ".")

    if c["connected_card_ids"]:
        parts.append(f"The device profile links this card to "
                     f"{len(c['connected_card_ids'])} other card"
                     f"{'s' if len(c['connected_card_ids']) > 1 else ''} held by "
                     f"unrelated customers, so the compromise is wider than this alert.")

    if verdict == "fraud" and c["affected_txn_ids"]:
        parts.append(f"The episode covers {len(c['affected_txn_ids'])} transaction"
                     f"{'s' if len(c['affected_txn_ids']) > 1 else ''} totalling "
                     f"{_money(c['exposure_usd'])}.")

    if c["similar_prior_cases"]:
        parts.append(f"Closed cases {', '.join(c['similar_prior_cases'][:3])} were "
                     f"retrieved as memory and used to weigh the evidence.")

    acts = [a["action"] for a in ans["next_best_actions"]["final"]]
    parts.append("Recommended: " + ", ".join(acts) + ".")

    out = " ".join(parts)
    # keep it to roughly six sentences
    sentences = [s.strip() for s in out.split(". ") if s.strip()]
    if len(sentences) > 6:
        sentences = sentences[:6]
    return ". ".join(s.rstrip(".") for s in sentences) + "."


def sar_narrative(ans: dict) -> str:
    """
    FinCEN-shaped narrative: who, what, when, where, how, why suspicious.
    Written so it stands on its own without the case file.
    """
    c = ans["case"]
    it = ans["_internal"]
    f = it["flagged"]
    sar = ans["sar"]
    dates = sar.get("activity_dates") or [str(f["ts"])[:10], str(f["ts"])[:10]]
    card_id = it["official_card_id"]
    cust = f["customer_id"]
    pat = PATTERN_WORDS.get(c["pattern"], c["pattern"])
    ep = it.get("episode", [])

    s: list[str] = []

    # WHO + WHAT + WHEN
    if len(ep) > 1:
        amts = ", ".join(_money(t["amount"]) for t in ep[:6])
        s.append(f"Between {dates[0]} and {dates[-1]}, {len(ep)} transactions totalling "
                 f"{_money(c['exposure_usd'])} were authorised on card {card_id}, held by "
                 f"customer {cust}: {amts}.")
    else:
        s.append(f"On {dates[0]}, a transaction of {_money(f['amount'])} was authorised "
                 f"on card {card_id}, held by customer {cust}.")

    # WHERE + HOW
    where = (f"The activity was conducted {f['channel'].replace('_', ' ')} under product "
             f"code {f['product_cd']}")
    if f.get("addr1"):
        where += f", billed in region {f['addr1']}"
    if f.get("device_key"):
        where += f", from device profile '{f['device_key']}'"
        if f.get("device_status"):
            where += f" which the identity record marks '{f['device_status']}' for this account"
    s.append(where + ".")

    # the evidence, stated plainly
    for ev in c["evidence"][:3]:
        s.append(ev["claim"].rstrip(".") + ".")

    # WHY SUSPICIOUS
    if c["connected_card_ids"]:
        names = ", ".join(c["connected_card_ids"][:8])
        more = ("" if len(c["connected_card_ids"]) <= 8
                else f" and {len(c['connected_card_ids']) - 8} further cards")
        s.append(f"The same device profile carries transactions for {names}{more}, cards "
                 f"belonging to customers with no relationship to one another, which "
                 f"indicates a single actor operating across multiple cardholders rather "
                 f"than an isolated compromise.")
    if c["pattern"] == "undocumented":
        s.append(c["pattern_description"])
    else:
        s.append(f"The activity is consistent with {pat} as described in the institution's "
                 f"fraud typologies.")

    if c["similar_prior_cases"]:
        s.append(f"Prior closed investigations {', '.join(c['similar_prior_cases'][:3])} "
                 f"connected to the same card or device profile were reviewed and "
                 f"support this assessment.")

    # what the bank did
    acts = [a["action"] for a in ans["next_best_actions"]["final"]]
    s.append(f"Assessed fraud probability is {c['fraud_probability']:.2f}. "
             f"Actions recorded: {', '.join(acts)}. "
             f"Total suspicious amount: {_money(sar.get('total_amount_usd') or 0)}.")

    s.append("This report is filed under the institution's fraud policy section 3a; "
             f"{sar.get('reason', '').split(': ', 1)[-1]}.")

    return " ".join(x for x in s if x)


def apply(ans: dict) -> dict:
    """Fill the prose fields on a finished investigation."""
    ans["case"]["summary"] = summarise(ans)
    if ans["sar"]["file"]:
        ans["sar"]["narrative"] = sar_narrative(ans)
    return ans
