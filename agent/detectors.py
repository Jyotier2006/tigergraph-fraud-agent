"""
Deterministic fraud-pattern detectors.

Each detector consumes graph evidence (never raw dataframes) and returns a
``Signal``: a named, weighted, quotable finding.  The signals are what the
LLM reasons over and what ends up in ``case.evidence`` -- so every number in a
case file is traceable to a graph query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


@dataclass
class Signal:
    name: str
    fired: bool
    weight: float                 # log-odds contribution when fired
    claim: str = ""
    source: str = "graph"
    ref: str = ""
    entity_ids: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


RARE_PROFILE_CARDS = 60    # above this a 'device profile' is a browser bucket,
                           # not a machine: 'Windows | Windows 10 | chrome 63.0
                           # | 1920x1080' alone is shared by 842 cards.

SMALL_AUTH = 5.00          # "tiny authorisation" ceiling (policy pattern 1)
TEST_WINDOW_MIN = 60       # card-testing clustering window
BIG_AFTER_TEST = 50.00     # "larger purchase" floor


def card_testing(window: dict, flagged: dict, hist: dict) -> Signal:
    """Pattern 1 / rule R5: >=3 tiny online auths within an hour, then a larger buy."""
    ref = window.get("ref", "")
    txns = [t for t in window.get("txns", [])]
    small = [t for t in txns if t["channel"] == "online" and (t["amount"] or 0) <= SMALL_AUTH]
    if len(small) < 3:
        return Signal("card_testing", False, 0.0, ref=ref)

    small.sort(key=lambda t: t["ts"])
    best = []
    for i, a in enumerate(small):
        ta = pd.Timestamp(a["ts"])
        grp = [b for b in small[i:]
               if (pd.Timestamp(b["ts"]) - ta).total_seconds() <= TEST_WINDOW_MIN * 60]
        if len(grp) > len(best):
            best = grp
    if len(best) < 3:
        return Signal("card_testing", False, 0.0, ref=ref)

    t_end = pd.Timestamp(best[-1]["ts"])
    follow = [t for t in txns
              if pd.Timestamp(t["ts"]) > t_end
              and (pd.Timestamp(t["ts"]) - t_end).total_seconds() <= 6 * 3600
              and (t["amount"] or 0) >= BIG_AFTER_TEST]
    ids = [t["txn_id"] for t in best] + [t["txn_id"] for t in follow]
    amts = ", ".join(f"${t['amount']:.2f}" for t in best[:5])
    span = (t_end - pd.Timestamp(best[0]["ts"])).total_seconds() / 60.0
    claim = (f"{len(best)} online authorisations of {amts} on this card within "
             f"{span:.0f} minutes")
    if follow:
        claim += (f", followed within {(pd.Timestamp(follow[0]['ts']) - t_end).total_seconds()/60:.0f}"
                  f" minutes by a ${follow[0]['amount']:.2f} purchase")
    return Signal("card_testing", True, 2.6 if follow else 1.9, claim=claim, ref=ref,
                  entity_ids=ids,
                  detail={"small_ids": [t["txn_id"] for t in best],
                          "followup_ids": [t["txn_id"] for t in follow],
                          "cleared_over_100": any((t["amount"] or 0) > 100 for t in follow)})


def amount_anomaly(flagged: dict, hist: dict) -> Signal:
    amt = flagged.get("amount") or 0.0
    med = hist.get("amount_median") or 0.0
    p95 = hist.get("amount_p95") or 0.0
    mx = hist.get("amount_max") or 0.0
    if not med:
        return Signal("amount_anomaly", False, 0.0, ref=hist.get("ref", ""))
    ratio = amt / med if med else 0
    fired = amt > max(p95, 1.0) and ratio >= 3.0
    claim = (f"Flagged amount ${amt:,.2f} is {ratio:.1f}x this card's median "
             f"${med:,.2f} (95th percentile ${p95:,.2f}, prior maximum ${mx:,.2f})")
    w = 0.9 if amt > mx else 0.5
    return Signal("amount_anomaly", fired, w, claim=claim, ref=hist.get("ref", ""),
                  entity_ids=[flagged["txn_id"]],
                  detail={"ratio": round(ratio, 2), "median": med, "p95": p95, "max": mx})


def product_novelty(flagged: dict, hist: dict) -> Signal:
    pc = flagged.get("product_cd")
    seen = hist.get("product_codes", {}) or {}
    fired = pc not in seen
    claim = (f"Product code {pc} has never appeared on this card "
             f"(history uses {', '.join(sorted(seen)) or 'none'})" if fired else
             f"Product code {pc} is one the cardholder already uses "
             f"({seen.get(pc, 0)} prior transactions)")
    return Signal("product_novelty", fired, 0.7, claim=claim, ref=hist.get("ref", ""),
                  entity_ids=[flagged["txn_id"]], detail={"seen": seen})


def channel_novelty(flagged: dict, hist: dict) -> Signal:
    ch = flagged.get("channel")
    seen = hist.get("channels", {}) or {}
    n_prior = seen.get(ch, 0)
    fired = n_prior == 0 and sum(seen.values()) > 20
    claim = (f"This card had never transacted {ch} before "
             f"({', '.join(f'{k}: {v}' for k, v in seen.items())})" if fired else
             f"Channel {ch} is normal for this card ({n_prior} prior transactions)")
    return Signal("channel_novelty", fired, 0.8, claim=claim, ref=hist.get("ref", ""),
                  entity_ids=[flagged["txn_id"]], detail={"channels": seen})


def new_device(flagged: dict, hist: dict, dev: dict) -> Signal:
    dk = flagged.get("device_key")
    if not dk:
        return Signal("new_device", False, 0.0, ref=hist.get("ref", ""))
    known = set(hist.get("device_keys", []) or [])
    status = (flagged.get("device_status") or "").strip()
    fired = (dk not in known) or status == "New"
    bits = []
    if dk not in known:
        bits.append("device profile not seen on this card before")
    if status:
        bits.append(f"identity record marks the device '{status}' for this account")
    proxy = (flagged.get("proxy_status") or "").strip()
    if proxy and proxy.lower() not in ("transparent", "nan"):
        bits.append(f"connection behind a {proxy} proxy")
    claim = f"Device profile '{dk}': " + "; ".join(bits) if bits else f"Device profile '{dk}' is known to this card"
    w = 1.2 if (dk not in known and status == "New") else 0.9
    if proxy and proxy.lower() not in ("transparent", "nan", ""):
        w += 0.5
    return Signal("new_device", fired, w, claim=claim, ref=hist.get("ref", ""),
                  entity_ids=[flagged["txn_id"]],
                  detail={"device_key": dk, "device_status": status, "proxy": proxy})


def shared_device(dev: dict, phys_card: str) -> Signal:
    """Rule R6: one device profile touching several cards is a shared origin."""
    if not dev.get("found"):
        return Signal("shared_device", False, 0.0, ref=dev.get("ref", ""))
    others = [c for c in dev.get("cards", []) if c["phys_card"] != phys_card]
    n_total = dev.get("n_cards_total", 0)
    # only a *specific* profile identifies a machine; a generic browser bucket
    # shared by hundreds of cardholders is not a shared origin.
    specific = dev.get("device_specific", False)
    fired = bool(specific and len(others) >= 1 and n_total <= RARE_PROFILE_CARDS)
    names = ", ".join(c["card_id"] for c in others[:6])
    claim = (f"Device profile '{dev['device_key']}' was used by {len(others)} other "
             f"card(s) in the surrounding window ({names}); {n_total} distinct cards "
             f"have used it across the dataset")
    w = 1.6 if len(others) >= 2 else 1.0
    return Signal("shared_device", fired, w, claim=claim, ref=dev.get("ref", ""),
                  entity_ids=[c["card_id"] for c in others[:10]],
                  detail={"other_cards": others, "n_cards_total": n_total})


def device_ring(dev: dict, phys_card: str) -> Signal:
    """
    Undocumented pattern: a *rare, specific* device profile whose cards nearly
    all appear inside one short window -- several unrelated cardholders
    transacting from one machine fingerprint.

    Generic profiles (unknown device and unknown screen, shared by hundreds of
    cardholders) are excluded: they are a browser-family bucket, not a device.
    """
    if not dev.get("found"):
        return Signal("device_ring", False, 0.0, ref=dev.get("ref", ""))
    n_win = dev.get("n_cards_window", 0)
    n_tot = dev.get("n_cards_total", 0) or 1
    conc = dev.get("concentration", 0.0)
    specific = dev.get("device_specific", False)
    others = [c for c in dev.get("cards", []) if c["phys_card"] != phys_card]

    fired = bool(specific and n_win >= 3 and n_tot <= RARE_PROFILE_CARDS
                 and conc >= 0.45)
    if not fired:
        return Signal("device_ring", False, 0.0, ref=dev.get("ref", ""),
                      detail={"n_cards_window": n_win, "n_cards_total": n_tot,
                              "concentration": conc, "specific": specific})

    prods = dev.get("window_product_codes", {})
    rs = dev.get("window_risk_median")
    claim = (f"Device profile '{dev['device_key']}' carries {n_win} distinct cards "
             f"across unrelated customers inside a {dev['window_days']}-day window "
             f"({n_tot} cards on this profile in the whole dataset, "
             f"{conc:.0%} of them in this window). "
             f"Transactions cluster on product code(s) "
             f"{', '.join(f'{k} x{v}' for k, v in list(prods.items())[:3])}"
             + (f" and the bank's model scored them a median of {rs:.2f}, "
                f"so the cluster is invisible to the score." if rs is not None else "."))
    w = 2.2 if n_win >= 8 else 1.6
    return Signal("device_ring", True, w, claim=claim, ref=dev.get("ref", ""),
                  entity_ids=[c["card_id"] for c in others[:25]],
                  detail={"n_cards_window": n_win, "n_cards_total": n_tot,
                          "concentration": conc,
                          "other_cards": [c["card_id"] for c in others],
                          "risk_median": rs})


def out_of_region(flagged: dict, hist: dict, window: dict) -> Signal:
    """Pattern 4: card-present use in a region the cardholder has no history in."""
    a1 = flagged.get("addr1")
    if a1 is None or flagged.get("channel") != "in_person":
        return Signal("out_of_region", False, 0.0, ref=hist.get("ref", ""))
    regions = {float(k): v for k, v in (hist.get("regions", {}) or {}).items()
               if k not in ("nan", "None")}
    prior_here = regions.get(float(a1), 0)
    if prior_here > 0:
        return Signal("out_of_region", False, 0.0, ref=hist.get("ref", ""),
                      claim=f"Billing region {a1} is the cardholder's own "
                            f"({prior_here} prior transactions)")
    home = max(regions, key=regions.get) if regions else None
    w_txns = window.get("txns", [])
    here = [t for t in w_txns if t.get("addr1") == a1]
    home_during = [t for t in w_txns if home is not None and t.get("addr1") == home]
    days = len({str(t["ts"])[:10] for t in here})
    claim = (f"Card-present purchase in billing region {a1}, where this card has no "
             f"history (home region {home} carries {regions.get(home, 0)} transactions). "
             f"{len(here)} transaction(s) in the new region across {days} day(s)")
    if home_during:
        claim += (f", while {len(home_during)} transaction(s) continued in the home "
                  f"region inside the same window — consistent with a cloned card "
                  f"rather than travel")
    w = 1.7 if home_during else (0.6 if days >= 3 else 1.1)
    return Signal("out_of_region", True, w, claim=claim, ref=hist.get("ref", ""),
                  entity_ids=[t["txn_id"] for t in here] or [flagged["txn_id"]],
                  detail={"region": a1, "home": home, "days_in_region": days,
                          "home_activity_during": len(home_during),
                          "region_txn_ids": [t["txn_id"] for t in here]})


def match_flag_anomaly(flagged: dict) -> Signal:
    mf = flagged.get("match_flags", {}) or {}
    false_flags = [k for k, v in mf.items() if v == "F"]
    fired = len(false_flags) >= 2
    claim = (f"Identity match flags {', '.join(false_flags)} are 'F' on the flagged "
             f"transaction (Vesta match checks such as name-to-address agreement); "
             f"these are unnamed model features, used here only as corroboration")
    return Signal("match_flag_anomaly", fired, 0.45, claim=claim,
                  ref="query:get_transaction", entity_ids=[flagged["txn_id"]],
                  detail={"false_flags": false_flags})


def email_novelty(flagged: dict, hist: dict) -> Signal:
    dom = flagged.get("p_email")
    seen = hist.get("email_domains", {}) or {}
    if not dom:
        return Signal("email_novelty", False, 0.0, ref=hist.get("ref", ""))
    fired = dom not in seen and sum(seen.values()) > 10
    claim = (f"Purchaser email domain '{dom}' is new to this card "
             f"(previously {', '.join(list(seen)[:4])})" if fired else
             f"Purchaser email domain '{dom}' matches the cardholder's history")
    return Signal("email_novelty", fired, 0.5, claim=claim, ref=hist.get("ref", ""),
                  entity_ids=[flagged["txn_id"]], detail={"domain": dom})


def burst(window: dict, flagged: dict, hist: dict) -> Signal:
    """Pattern 2: two to four online transactions inside 48 hours."""
    txns = [t for t in window.get("txns", []) if t["channel"] == "online"]
    if len(txns) < 2:
        return Signal("burst", False, 0.0, ref=window.get("ref", ""))
    ft = pd.Timestamp(flagged["ts"])
    near = [t for t in txns
            if abs((pd.Timestamp(t["ts"]) - ft).total_seconds()) <= 48 * 3600]
    known_products = set(hist.get("product_codes", {}) or {})
    med = hist.get("amount_median") or 0
    odd = [t for t in near
           if t["product_cd"] not in known_products or (t["amount"] or 0) > 3 * med]
    fired = 2 <= len(near) <= 8 and len(odd) >= 2
    total = sum(t["amount"] or 0 for t in odd)
    claim = (f"{len(near)} online transactions on this card within 48 hours, "
             f"{len(odd)} of them outside the cardholder's product or amount profile, "
             f"totalling ${total:,.2f}")
    return Signal("burst", fired, 0.9, claim=claim, ref=window.get("ref", ""),
                  entity_ids=[t["txn_id"] for t in odd],
                  detail={"odd_ids": [t["txn_id"] for t in odd], "n_near": len(near)})


def prior_case_signal(mem: dict, phys_card: str) -> Signal:
    cases = mem.get("cases", [])
    if not cases:
        return Signal("prior_cases", False, 0.0, ref=mem.get("ref", ""))
    fraud = [c for c in cases if c["outcome"] == "confirmed_fraud"]
    cleared = [c for c in cases if c["outcome"] == "cleared"]
    # Only same-card links (and same-device links through a *rare* profile,
    # which similar_prior_cases already filters) count as strong memory. A
    # shared generic browser bucket would otherwise link every alert to
    # hundreds of unrelated closed cases.
    strong = [c for c in fraud if "same card" in c["why"] or
              "same device profile" in c["why"]]
    weak_cleared = [c for c in cleared if "same card" in c["why"] or
                    "same device profile" in c["why"]]
    if strong:
        c = strong[0]
        claim = (f"Closed case {c['case_id']} ({c['pattern']}, confirmed fraud, "
                 f"${c['exposure_usd']:,.2f}) is linked to this alert by "
                 f"{', '.join(c['why'])}: \"{c['analyst_notes'][:180]}\"")
        return Signal("prior_cases", True, 1.3, claim=claim, ref=mem.get("ref", ""),
                      entity_ids=[c["case_id"] for c in strong[:4]],
                      detail={"linked": [c["case_id"] for c in strong]})
    if weak_cleared:
        c = weak_cleared[0]
        claim = (f"Closed case {c['case_id']} on the same card/device was cleared as a "
                 f"false alarm: \"{c['analyst_notes'][:180]}\"")
        return Signal("prior_cases", True, -0.9, claim=claim, ref=mem.get("ref", ""),
                      entity_ids=[c["case_id"] for c in weak_cleared[:4]],
                      detail={"linked": [c["case_id"] for c in weak_cleared]})
    return Signal("prior_cases", False, 0.0, ref=mem.get("ref", ""))


def recurring_charge(rec: dict, flagged: dict) -> Signal:
    """Rule R7: a disputed charge that matches the cardholder's own recurring pattern."""
    n = rec.get("n_prior_same_amount", 0)
    monthly = bool(rec.get("looks_monthly"))
    established = bool(rec.get("established_repeat"))
    fired = established or (n >= 2 and monthly)
    if not fired:
        return Signal("recurring_charge", False, 0.0, ref=rec.get("ref", ""), detail=rec)
    shape = ("on a roughly monthly cycle" if monthly else
             f"repeatedly over {rec.get('span_days', 0):.0f} days")
    claim = (f"The disputed ${flagged['amount']:,.2f} charge is one this card has "
             f"already made {n} times under the same product code, {shape} "
             f"(most recently {str(rec.get('last_prior_ts'))[:10]}). The dataset has no "
             f"merchant field, so same-amount-and-product is used as the merchant "
             f"proxy for policy R7; the charge matches the cardholder's own "
             f"established pattern rather than standing outside it")
    w = -2.6 if n >= 20 else (-2.2 if n >= 5 else -1.8)
    return Signal("recurring_charge", True, w, claim=claim, ref=rec.get("ref", ""),
                  entity_ids=rec.get("prior_txn_ids", []), detail=rec)


def risk_score_signal(flagged: dict) -> Signal:
    rs = flagged.get("risk_score") or 0.0
    if rs >= 0.85:
        w, fired = 0.45, True
    elif rs >= 0.70:
        w, fired = 0.2, True
    elif rs <= 0.30:
        w, fired = -0.25, True
    else:
        w, fired = 0.0, False
    claim = (f"The bank's model scored this transaction {rs:.2f}. Per the dataset "
             f"documentation most transactions scored above 0.70 are legitimate, so "
             f"the score is treated as a reason to look, not as evidence of fraud")
    return Signal("risk_score", fired, w, claim=claim, source="document",
                  ref="policy:Section 0 / README 'Things to know'",
                  entity_ids=[flagged["txn_id"]], detail={"risk_score": rs})


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
