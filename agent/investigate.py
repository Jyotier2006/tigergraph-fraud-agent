"""
The investigation agent.

A bounded, step-wise loop:

    1 trigger      -> read the alert
    2 anchor       -> fetch the flagged transaction from the graph
    3 baseline     -> the card's own behavioural profile
    4 window       -> what happened around the alert on this card
    5 spread       -> device profile / region traversal to other cards
    6 memory       -> retrieve closed cases linked by card, device or pattern
    7 assess       -> detectors -> log-odds -> pattern, probability, episode
    8 uncertainty  -> policy R1/R8: is more evidence required before acting?
    9 re-assess    -> fold the (simulated) response back in
   10 decide       -> initial and final next-best-actions, SAR test
   11 remember     -> write the case to the graph

The agent stops as soon as policy section 6 is satisfied; `stop_reason` records
which clause fired.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import detectors as D
from . import policy as P
from .tools import LocalBackend, get_backend

PATTERNS = {"card_testing", "card_not_present_fraud", "card_not_present_new_device",
            "out_of_region_use", "account_takeover", "undocumented", "none"}

_W = None


def _weights() -> dict:
    global _W
    if _W is None:
        import json
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "weights.json")) as fh:
            _W = json.load(fh)
    return _W


# Policy-derived adjustments the closed-case file cannot teach.
#
#  * A customer report IS the cardholder denying the charge (policy R2); the
#    closed-case history does not record which alerts came from customers, so
#    the denial is added as an explicit prior rather than learned.
#  * Card testing is a documented typology with its own rule (R5) and only 16
#    examples in the closed file, far too few to learn a weight from.
#  * A device profile shared across several cards is the R6 shared-origin
#    trigger; it is rare in history but is decisive when present.
OVERRIDES = {
    "customer_report_denial": 1.40,
    "card_testing": 2.30,
    "device_ring_large": 2.10,
    "device_ring": 1.55,
    "out_of_region_clone": 1.45,
    "recurring_charge": -2.40,
}

# Coefficients learned on model-triggered alerts that must not be transplanted
# onto a customer dispute.  "Device marked New" is the single strongest
# exonerating feature among false-alarm alerts (the cardholder bought a phone)
# -- but when the cardholder is the one saying the charge is not theirs, a
# device new to the account corroborates the dispute instead of explaining it.
SUPPRESS_FOR_DISPUTE = {"device_status_new", "hist_thin", "night_hour",
                        "amount_ratio_log", "amount_over_max"}


def score_log_odds(feats: dict, trigger: str, fired: dict) -> tuple[float, list]:
    """
    Calibrated log-odds plus the rule-driven overrides.

    The logistic part is trained on the bank's closed alerts, so it is applied
    in the regime it was learned in.  Anything the closed cases cannot teach --
    the documented typologies with too few historical examples, and the
    undocumented device ring the graph finds structurally -- is added as an
    explicit, cited override rather than pretended to be learned.
    """
    w = _weights()
    lo = float(w["intercept"])
    notes = [("calibrated intercept", round(lo, 3))]
    dispute = trigger == "customer_report"

    for name, coef in w["coef"].items():
        v = float(feats.get(name, 0.0))
        if not v or not coef:
            continue
        if dispute and name in SUPPRESS_FOR_DISPUTE:
            notes.append((f"{name} suppressed (learned on model-triggered alerts, "
                          f"not on disputes)", 0.0))
            continue
        # the learned sign on out-of-region contradicts documented pattern 4;
        # the typology is authoritative and is applied as an override below.
        if name.startswith("out_of_region"):
            continue
        lo += coef * v
        notes.append((f"{name}={v:g}", round(coef * v, 3)))

    if dispute and "recurring_charge" not in fired:
        lo += OVERRIDES["customer_report_denial"]
        notes.append(("cardholder denies the charge (R2)",
                      OVERRIDES["customer_report_denial"]))
    if "card_testing" in fired:
        lo += OVERRIDES["card_testing"]
        notes.append(("documented pattern 1: card testing (R5)",
                      OVERRIDES["card_testing"]))
    if "recurring_charge" in fired:
        bump = fired["recurring_charge"].weight
        lo += bump
        notes.append(("charge matches the card's own established repeat pattern (R7)",
                      bump))
    ring = fired.get("device_ring")
    if ring is not None:
        big = ring.detail.get("n_cards_window", 0) >= 8
        bump = OVERRIDES["device_ring_large"] if big else OVERRIDES["device_ring"]
        lo += bump
        notes.append(("rare device profile shared across unrelated cards (R6/R9)", bump))
    oor = fired.get("out_of_region")
    if oor is not None:
        if oor.detail.get("home_activity_during", 0) > 0:
            lo += OVERRIDES["out_of_region_clone"]
            notes.append(("documented pattern 4: card-present use in a new region "
                          "while home activity continues", OVERRIDES["out_of_region_clone"]))
        elif oor.detail.get("days_in_region", 0) >= 3:
            lo -= 0.70
            notes.append(("several days of purchases in one new region reads as "
                          "travel, not a clone", -0.70))
        else:
            lo += 0.55
            notes.append(("card-present use in a region with no history", 0.55))
    return lo, notes


def _mk_evidence(sig: D.Signal) -> dict:
    return {"claim": sig.claim, "source": sig.source, "ref": sig.ref,
            "entity_ids": [str(e) for e in sig.entity_ids]}


@dataclass
class Investigation:
    case_row: pd.Series
    backend: LocalBackend
    steps: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    signals: list = field(default_factory=list)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        t0 = time.time()
        c = self.case_row
        be = self.backend

        # --- 1 trigger ---------------------------------------------------
        trigger = str(c["trigger_type"])
        self.steps.append(f"1 trigger: {trigger}")

        # --- 2 anchor ----------------------------------------------------
        flagged = be.get_transaction(int(c["flagged_txn_id"]))
        if not flagged.get("found"):
            raise RuntimeError(f"flagged transaction {c['flagged_txn_id']} not in dataset")
        phys = flagged["phys_card"]
        ts = pd.Timestamp(flagged["ts"])
        official_card_id = str(c["card_id"])
        self.steps.append(f"2 anchor: txn {flagged['txn_id']} ${flagged['amount']:,.2f} "
                          f"{flagged['channel']} at {flagged['ts']}")

        # --- 3 baseline --------------------------------------------------
        hist = be.card_history(phys, before_ts=ts)
        self.steps.append(f"3 baseline: {hist['n_txns']} prior transactions on the card")

        # --- 4 window ----------------------------------------------------
        win = be.card_window(phys, ts, hours=72)
        self.steps.append(f"4 window: {win['n']} transactions within +/-72h")

        # --- 5 spread ----------------------------------------------------
        dev = be.device_neighbors(flagged.get("device_key") or "", ts, days=30)
        region = be.region_cohort(flagged.get("addr1"), ts, days=14)
        cust_cards = be.customer_cards(flagged["customer_id"])
        self.steps.append(
            f"5 spread: device profile -> {len(dev.get('cards', []))} card(s); "
            f"customer holds {cust_cards['n_cards']} card(s)")

        # --- 6 memory ----------------------------------------------------
        dev_keys = [flagged["device_key"]] if flagged.get("device_key") else []
        mem = be.similar_prior_cases(phys, dev_keys, flagged["customer_id"])
        self.steps.append(f"6 memory: retrieved {len(mem['cases'])} closed case(s)")

        # --- 7 assess ----------------------------------------------------
        rec = be.recurring_match(phys, flagged["amount"], flagged["product_cd"], ts)

        sigs = [
            D.card_testing(win, flagged, hist),
            D.out_of_region(flagged, hist, win),
            D.new_device(flagged, hist, dev),
            D.shared_device(dev, phys),
            D.device_ring(dev, phys),
            D.amount_anomaly(flagged, hist),
            D.product_novelty(flagged, hist),
            D.channel_novelty(flagged, hist),
            D.email_novelty(flagged, hist),
            D.burst(win, flagged, hist),
            D.match_flag_anomaly(flagged),
            D.prior_case_signal(mem, phys),
            D.risk_score_signal(flagged),
        ]
        if trigger == "customer_report":
            sigs.append(D.recurring_charge(rec, flagged))
        self.signals = sigs
        fired = {s.name: s for s in sigs if s.fired}

        # --- scoring: calibrated model + documented-pattern overrides -----
        from .features import collect as _collect
        feats = _collect(be, flagged, trigger, official_card_id)["features"]
        lo, notes = score_log_odds(feats, trigger, fired)
        prob = float(np.clip(D.sigmoid(lo), 0.03, 0.94))

        pattern, pattern_desc = self._classify(fired, flagged, hist, dev, trigger)
        episode = self._episode(fired, flagged, win, pattern)
        exposure = round(sum(t["amount"] or 0 for t in episode), 2)

        # evidence list (only what fired, plus the framing of the risk score)
        order = ["device_ring", "card_testing", "out_of_region", "shared_device", "new_device",
                 "recurring_charge", "prior_cases", "burst", "amount_anomaly",
                 "product_novelty", "channel_novelty", "email_novelty",
                 "match_flag_anomaly", "risk_score"]
        for name in order:
            s = fired.get(name)
            if s and s.claim:
                self.evidence.append(_mk_evidence(s))

        if "device_ring" in fired or "shared_device" in fired:
            connected_cards = sorted({c0["card_id"] for c0 in dev.get("cards", [])
                                      if c0["phys_card"] != phys})
        else:
            connected_cards = []
        # GraphRAG: retrieve the policy / typology text this case actually turns
        # on and cite it, rather than asserting a rule from memory.
        from .graphrag import policy_index
        rag_q = " ".join([pattern.replace("_", " ")] +
                         [s0.claim for s0 in sigs if s0.fired][:3] +
                         [trigger.replace("_", " ")])
        for chunk in policy_index().retrieve(rag_q, k=2):
            self.evidence.append({
                "claim": chunk["text"][:420].replace("\n", " ").strip(),
                "source": "document",
                "ref": (f"policy:{chunk['rule_id']} ({chunk['section']})"
                        if chunk["rule_id"] else
                        f"document:{chunk['doc']} / {chunk['section']}"),
                "entity_ids": [],
            })

        connected_devices = [flagged["device_key"]] if (
            flagged.get("device_key") and connected_cards
        ) else []
        shared_origin = ("shared_device" in fired) or ("device_ring" in fired)
        connected_fraud = bool(fired.get("prior_cases") and
                               fired["prior_cases"].weight > 0 and connected_cards)

        n_independent = len([s for s in sigs if s.fired and s.weight > 0.4
                             and s.name != "risk_score"])
        self.steps.append(f"7 assess: pattern={pattern} p={prob:.2f} "
                          f"independent-signals={n_independent}")

        # --- 8 uncertainty / evidence requests ---------------------------
        ev_requests, initial_actions, assumed_effect = self._evidence_stage(
            trigger, prob, pattern, fired, exposure, connected_cards, rec)

        # --- 9 re-assess --------------------------------------------------
        prob_final = float(np.clip(D.sigmoid(lo + assumed_effect), 0.03, 0.95))
        settled = bool(ev_requests)
        verdict = ("fraud" if prob_final >= 0.70 else
                   "legitimate" if prob_final <= 0.30 else "uncertain")
        if verdict == "legitimate":
            episode, exposure = [], 0.0
        elif pattern == "none":
            # a fraud or uncertain verdict must name what it thinks is happening
            pattern, pattern_desc = self._fallback_pattern(flagged, fired, hist)
        self.steps.append(f"9 re-assess: p={prob_final:.2f} verdict={verdict}")

        # --- 10 decide ----------------------------------------------------
        final_actions, sar = self._decide(
            trigger, verdict, prob_final, pattern, exposure, fired,
            connected_cards, shared_origin, connected_fraud, cust_cards,
            flagged, episode, official_card_id)

        status = ("closed_fraud" if verdict == "fraud" and not any(
                      a["action"] == "ESCALATE_TO_ANALYST" for a in final_actions)
                  else "escalated" if any(a["action"] == "ESCALATE_TO_ANALYST"
                                          for a in final_actions)
                  else "closed_legitimate" if verdict == "legitimate" else "open")

        what_changed = ("nothing" if not ev_requests else
                        f"The assumed response to the {ev_requests[0]['type'].replace('_', ' ')} "
                        f"moved fraud probability from {prob:.2f} to {prob_final:.2f}"
                        + (", which changes the recommended actions under the policy."
                           if final_actions != initial_actions else
                           ", confirming the initial recommendation."))

        if not ev_requests:
            initial_actions = [dict(a) for a in final_actions]

        self.steps.append(f"10 decide: {[a['action'] for a in final_actions]}")

        case = {
            "status": status,
            "verdict": verdict,
            "fraud_probability": round(prob_final, 2),
            "pattern": (pattern if verdict != "legitimate" else "none"),
            "pattern_description": pattern_desc if pattern == "undocumented" else "",
            "affected_txn_ids": [t["txn_id"] for t in episode],
            "first_suspicious_txn_id": episode[0]["txn_id"] if episode else "",
            "connected_card_ids": connected_cards,
            "connected_device_profiles": connected_devices,
            "exposure_usd": exposure,
            "evidence": self.evidence,
            "similar_prior_cases": [c0["case_id"] for c0 in mem["cases"][:5]],
            "summary": "",                      # filled by the narrator
            "written_to_graph": False,
            "graph_case_id": "",
        }

        return {
            "case_id": str(c["case_id"]),
            "case": case,
            "evidence_requests": ev_requests,
            "next_best_actions": {"initial": initial_actions, "final": final_actions,
                                  "what_changed": what_changed},
            "sar": sar,
            "stop_reason": P.stop_reason(prob_final, n_independent, settled, exposure),
            "tool_calls": be.tool_calls,
            "tokens": 0,
            "latency_s": round(time.time() - t0, 2),
            # internals used by the narrator / graph writer (stripped on output)
            "_internal": {
                "flagged": flagged, "hist": hist, "dev": dev, "mem": mem,
                "fired": sorted(fired), "log_odds": notes, "steps": self.steps,
                "prob_initial": round(prob, 2), "episode": episode,
                "official_card_id": official_card_id, "phys_card": phys,
                "region": region, "cust_cards": cust_cards,
                "n_independent": n_independent,
            },
        }

    # ------------------------------------------------------------ helpers
    def _classify(self, fired, flagged, hist, dev, trigger):
        desc = ""
        if "card_testing" in fired:
            return "card_testing", desc
        if "out_of_region" in fired and flagged["channel"] == "in_person":
            return "out_of_region_use", desc
        # A rare device profile carrying many unrelated cardholders is
        # coordinated abuse across customers: policy R9 says to describe it in
        # our own words rather than force it into one of the five typologies.
        if "device_ring" in fired:
            d = fired["device_ring"].detail
            n_win = d.get("n_cards_window", 0)
            n_tot = d.get("n_cards_total", 0)
            rs = d.get("risk_median")
            desc = (f"A single rare device profile ({dev['device_key']}) carries "
                    f"transactions for {n_win} different cards belonging to unrelated "
                    f"customers inside a {dev.get('window_days', 15)}-day window, "
                    f"{d.get('concentration', 0):.0%} of every card that profile has "
                    f"ever been seen with ({n_tot} in total). There is no testing "
                    f"sequence, no shared billing region and no single compromised "
                    f"cardholder: the common element is the machine. "
                    + (f"The bank's model scored the cluster a median of {rs:.2f}, so "
                       f"none of it was flagged individually and it is only visible by "
                       f"traversing the device profile out to its neighbouring cards. "
                       if rs is not None else "")
                    + f"It affects every cardholder on that device, and fits none of "
                    f"the five documented typologies.")
            return "undocumented", desc
        if flagged["channel"] == "online":
            if "new_device" in fired:
                return "card_not_present_new_device", desc
            if ("burst" in fired or "amount_anomaly" in fired or
                    "product_novelty" in fired or trigger == "customer_report"):
                return "card_not_present_fraud", desc
        if "channel_novelty" in fired and "match_flag_anomaly" in fired:
            return "account_takeover", desc
        if trigger == "customer_report":
            return "card_not_present_fraud", desc
        return "none", desc

    def _fallback_pattern(self, flagged, fired, hist):
        """A non-legitimate verdict must name a typology or describe its own."""
        if flagged["channel"] == "online":
            return ("card_not_present_new_device" if "new_device" in fired
                    else "card_not_present_fraud"), ""
        desc = (f"Card-present activity on this card that matches none of the five "
                f"documented typologies: the billing region is one the cardholder "
                f"already uses, there is no testing sequence and no device link, yet "
                f"the transaction sits outside the card's own amount and product "
                f"profile. It affects this cardholder only, and was found by "
                f"comparing the flagged transaction against the card's full history "
                f"rather than by matching a known pattern.")
        return "undocumented", desc

    def _episode(self, fired, flagged, win, pattern):
        """The set of transactions that make up the fraud episode."""
        by_id = {t["txn_id"]: t for t in win.get("txns", [])}
        ids: list[str] = []
        if "card_testing" in fired:
            d = fired["card_testing"].detail
            ids = list(d["small_ids"]) + list(d["followup_ids"])
        elif pattern == "out_of_region_use" and "out_of_region" in fired:
            ids = list(fired["out_of_region"].detail["region_txn_ids"])
        elif "device_ring" in fired or "shared_device" in fired:
            dk = flagged.get("device_key")
            ids = [t["txn_id"] for t in win.get("txns", [])
                   if t.get("device_key") == dk]
            if "burst" in fired:
                ids += list(fired["burst"].detail["odd_ids"])
        elif "burst" in fired:
            ids = list(fired["burst"].detail["odd_ids"])
        if flagged["txn_id"] not in ids:
            ids.append(flagged["txn_id"])
        seen, ep = set(), []
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            ep.append(by_id.get(i) or (flagged if i == flagged["txn_id"] else None))
        ep = [t for t in ep if t]
        ep.sort(key=lambda t: t["ts"])
        return ep

    def _evidence_stage(self, trigger, prob, pattern, fired, exposure,
                        connected_cards, rec):
        """
        Policy R1: on a single weak signal below 0.70, verify before blocking.
        The customer's reply is not provided, so it is simulated from the
        independent graph evidence and the assumption is recorded.
        """
        reqs, initial = [], []
        single_signal = len([s for s in self.signals
                             if s.fired and s.weight > 0.4 and s.name != "risk_score"]) <= 1

        if trigger == "customer_report":
            # the customer has already spoken: the dispute IS the statement
            if "recurring_charge" in fired:
                initial = [P.act("CREATE_CASE", "R7: disputed charge matches the "
                                 "cardholder's own recurring pattern; open a case to "
                                 "hold the evidence"),
                           P.act("VERIFY_WITH_CUSTOMER", "R7: confirm with the "
                                 "cardholder before any action on the card")]
                reqs = [{"type": "customer_validation", "asked_after_step": 7,
                         "assumed_response": (
                             "Simulated. Shown the earlier identical monthly charges, "
                             "the cardholder recognises the subscription and withdraws "
                             "the dispute. Assumption recorded per policy section 5; the "
                             "recurring match is the independent evidence behind it.")}]
                return reqs, initial, -1.60
            initial = [P.act("CREATE_CASE", "R2/3a: the cardholder disputes the charge, "
                             "so a case is opened to carry the evidence"),
                       P.act("MONITOR_CARD", "R2: raise monitoring while the "
                             "investigation confirms the scope of the compromise")]
            reqs = [{"type": "customer_validation", "asked_after_step": 7,
                     "assumed_response": (
                         "Simulated. The cardholder restates that they did not make the "
                         "transaction and still holds the card. Assumption recorded per "
                         "policy section 5; it is consistent with the graph evidence "
                         "above." if prob >= 0.45 else
                         "Simulated. On review with the cardholder the transaction is "
                         "recognised as their own. Assumption recorded per policy "
                         "section 5; the card's own history supports it.")}]
            return reqs, initial, (1.10 if prob >= 0.45 else -1.80)

        if "card_testing" in fired:
            d = fired["card_testing"].detail
            initial = [P.act("DECLINE_TRANSACTION",
                             "R5: a testing sequence is present on this card", exposure),
                       P.act("STEP_UP_AUTH", "R5: require step-up authentication before "
                             "any further activity on the card")]
            if d.get("cleared_over_100"):
                initial.insert(0, P.act("BLOCK_CARD", "R5: a purchase over $100 has "
                                        "already cleared after the testing sequence",
                                        exposure))
            reqs = [{"type": "step_up_auth", "asked_after_step": 8,
                     "assumed_response": (
                         "Simulated. The step-up challenge is not completed within the "
                         "policy window, which is consistent with the account being "
                         "operated by someone other than the cardholder. Assumption "
                         "recorded per policy section 5.")}]
            return reqs, initial, 0.60

        if prob < 0.70:
            # R1 -- verify before blocking on a weak or single signal
            why = ("R1: the case rests on a single signal and probability is "
                   f"{prob:.2f}, below 0.70" if single_signal else
                   f"R1: probability {prob:.2f} is below 0.70, so the cardholder is "
                   f"asked before anything that would affect the card")
            initial = [P.act("VERIFY_WITH_CUSTOMER", why),
                       P.act("MONITOR_CARD", "R1: keep the card active but raise "
                             "monitoring sensitivity while the answer is pending")]
            if pattern != "none" and prob >= 0.30:
                initial.append(P.act("CREATE_CASE", "3a: a case is opened because "
                                     "evidence has been requested"))
            lean_fraud = prob >= 0.45
            reqs = [{"type": "customer_validation", "asked_after_step": 8,
                     "assumed_response": (
                         "Simulated. The cardholder states they did not make this "
                         "transaction. Assumption recorded per policy section 5; it "
                         "follows the graph evidence above, which is independent of "
                         "the model score." if lean_fraud else
                         "Simulated. The cardholder confirms the transaction as their "
                         "own. Assumption recorded per policy section 5; it follows "
                         "the card's own history, which shows nothing out of pattern "
                         "beyond the model score.")}]
            return reqs, initial, (1.25 if lean_fraud else -1.60)

        # Nothing is asked: the graph evidence already satisfies policy section 6,
        # so the recommendation made before and after is the same one. The answer
        # format requires final == initial in exactly this case.
        return [], None, 0.0

    def _decide(self, trigger, verdict, prob, pattern, exposure, fired,
                connected_cards, shared_origin, connected_fraud, cust_cards,
                flagged, episode, official_card_id):
        acts: list[dict] = []

        if verdict == "legitimate":
            if trigger == "customer_report" and "recurring_charge" in fired:
                acts = [P.act("CREATE_CASE", "R7: the dispute and its resolution are "
                              "recorded against the card"),
                        P.act("WARN_CUSTOMER", "R7: send a recurring-charge reminder so "
                              "the same subscription is not disputed again"),
                        P.act("CLOSE_NO_FRAUD", "R3/R7: the charge matches the "
                              "cardholder's own recurring pattern")]
            else:
                acts = [P.act("ALLOW_TRANSACTION", "R3: the cardholder's own history "
                              "explains the transaction; nothing supports a decline"),
                        P.act("CLOSE_NO_FRAUD", "R3: the alert is a false positive of "
                              "the model score")]
            sar_file, sar_reason = P.sar_required(prob, verdict, 0.0, False, False, pattern)
            return acts, self._sar_block(sar_file, sar_reason, flagged, episode,
                                         connected_cards, official_card_id, pattern, 0.0)

        if verdict == "uncertain":
            acts = [P.act("CREATE_CASE", "3a: probability is at or above 0.30, so the "
                          "investigation is recorded as a case")]
            if exposure > 500:
                acts.append(P.act("ESCALATE_TO_ANALYST",
                                  f"R8: the verdict is uncertain and exposure "
                                  f"${exposure:,.2f} exceeds $500"))
            acts.append(P.act("MONITOR_CARD", "R4: keep the card active under raised "
                              "monitoring while a human reviews"))
            acts.append(P.act("STEP_UP_AUTH", "R1: require step-up authentication "
                              "before further activity rather than blocking on "
                              "inconclusive evidence"))
            if connected_cards:
                acts.append(P.act("MONITOR_CONNECTED_CARDS",
                                  f"R6: {len(connected_cards)} other card(s) share the "
                                  f"device profile named in the evidence"))
            sar_file, sar_reason = P.sar_required(prob, verdict, exposure, shared_origin,
                                                  connected_fraud, pattern)
            if sar_file:
                acts.append(P.act("FILE_REPORT", sar_reason, exposure))
            return acts, self._sar_block(sar_file, sar_reason, flagged, episode,
                                         connected_cards, official_card_id, pattern,
                                         exposure)

        # ---- verdict == fraud ------------------------------------------
        if "card_testing" in fired:
            acts.append(P.act("BLOCK_CARD", f"R5/R2: testing sequence confirmed and the "
                              f"card is compromised; exposure ${exposure:,.2f}", exposure))
        else:
            acts.append(P.act("BLOCK_CARD", f"R2: the cardholder denies the activity and "
                              f"the card is compromised; exposure ${exposure:,.2f}",
                              exposure))
        acts.append(P.act("CREATE_CASE", "R2: an internal case carries the evidence, the "
                          "affected transactions and the actions taken"))

        sar_file, sar_reason = P.sar_required(prob, verdict, exposure, shared_origin,
                                              connected_fraud, pattern)
        if sar_file:
            acts.append(P.act("FILE_REPORT", sar_reason, exposure))
        if connected_cards:
            acts.append(P.act("MONITOR_CONNECTED_CARDS",
                              f"R6: the device profile in the evidence is shared with "
                              f"{', '.join(connected_cards[:4])}"))
        if pattern == "undocumented":
            acts.append(P.act("ESCALATE_TO_ANALYST", "R9: the pattern matches none of "
                              "the five documented typologies and shows coordinated "
                              "abuse across customers"))
        # R10 guard
        n_cards = cust_cards.get("n_cards", 1)
        if n_cards > 1 and connected_fraud and len(
                [c for c in connected_cards if c.startswith(flagged["customer_id"])]) >= 1:
            acts.append(P.act("BLOCK_ALL_CARDS", "R10: a second card belonging to this "
                              "customer also shows confirmed fraud", exposure))
        return acts, self._sar_block(sar_file, sar_reason, flagged, episode,
                                     connected_cards, official_card_id, pattern, exposure)

    def _sar_block(self, file, reason, flagged, episode, connected_cards,
                   official_card_id, pattern, exposure):
        if not file:
            return {"file": False, "reason": reason, "narrative": "", "subjects": [],
                    "total_amount_usd": 0, "activity_dates": []}
        dates = sorted({str(t["ts"])[:10] for t in episode}) or [str(flagged["ts"])[:10]]
        subjects = [flagged["customer_id"], official_card_id] + connected_cards[:6]
        if flagged.get("device_key"):
            subjects.append(flagged["device_key"])
        return {"file": True, "reason": reason, "narrative": "",
                "subjects": subjects,
                "total_amount_usd": round(exposure, 2),
                "activity_dates": [dates[0], dates[-1]]}


def investigate(case_row, backend=None) -> dict:
    be = backend or get_backend("local")
    return Investigation(case_row, be).run()
