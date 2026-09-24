"""
Validate the answer files against the Answer Format in the dataset README.

    python tests/validate_answers.py

Checks structure, enums, cross-field consistency and -- the one that actually
bites -- that every id quoted in an answer exists in the dataset.  "Made-up IDs
score zero."
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import get_dataset   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "cases")

TOP = ["case_id", "case", "evidence_requests", "next_best_actions", "sar",
       "stop_reason", "tool_calls", "tokens", "latency_s"]
CASE = ["status", "verdict", "fraud_probability", "pattern", "pattern_description",
        "affected_txn_ids", "first_suspicious_txn_id", "connected_card_ids",
        "connected_device_profiles", "exposure_usd", "evidence",
        "similar_prior_cases", "summary", "written_to_graph", "graph_case_id"]
SAR = ["file", "reason", "narrative", "subjects", "total_amount_usd", "activity_dates"]

STATUS = {"open", "closed_fraud", "closed_legitimate", "escalated"}
VERDICT = {"fraud", "legitimate", "uncertain"}
PATTERN = {"card_testing", "card_not_present_fraud", "card_not_present_new_device",
           "out_of_region_use", "account_takeover", "undocumented", "none"}
SOURCE = {"graph", "document", "customer", "external"}
ROUTE = {"auto", "L1", "L2"}
EVREQ = {"customer_validation", "step_up_auth", "analyst_info"}
ACTIONS = {"ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD",
           "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER",
           "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS", "GENERATE_REPORT",
           "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD"}
AUTO = ACTIONS - {"DECLINE_TRANSACTION", "BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"}


def main() -> int:
    ds = get_dataset()
    valid_txn = {str(int(t)) for t in ds.txn["TransactionID"]}
    valid_card = set(ds.txn["card_id"]) | set(ds.pack["card_id"]) | set(ds.closed["card_id"].dropna())
    valid_case = set(ds.closed["case_id"])
    expected = set(ds.pack["case_id"])

    files = sorted(glob.glob(os.path.join(CASES, "HHG-*.json")))
    errors, warnings = [], []
    seen = set()

    for path in files:
        name = os.path.basename(path)
        with open(path) as fh:
            a = json.load(fh)
        cid = a.get("case_id", "?")
        seen.add(cid)
        E = lambda m: errors.append(f"{name}: {m}")          # noqa: E731
        W = lambda m: warnings.append(f"{name}: {m}")        # noqa: E731

        if os.path.splitext(name)[0] != cid:
            E(f"filename does not match case_id {cid}")
        for k in TOP:
            if k not in a:
                E(f"missing top-level field '{k}'")
        c = a.get("case", {})
        for k in CASE:
            if k not in c:
                E(f"case missing '{k}'")
        s = a.get("sar", {})
        for k in SAR:
            if k not in s:
                E(f"sar missing '{k}'")

        # ---- enums ---------------------------------------------------
        if c.get("status") not in STATUS:
            E(f"bad status {c.get('status')!r}")
        if c.get("verdict") not in VERDICT:
            E(f"bad verdict {c.get('verdict')!r}")
        if c.get("pattern") not in PATTERN:
            E(f"bad pattern {c.get('pattern')!r}")
        p = c.get("fraud_probability")
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            E(f"fraud_probability out of range: {p}")

        # ---- ids exist ------------------------------------------------
        for t in c.get("affected_txn_ids", []):
            if t not in valid_txn:
                E(f"affected txn {t} not in dataset")
        fs = c.get("first_suspicious_txn_id", "")
        if fs and fs not in valid_txn:
            E(f"first_suspicious_txn_id {fs} not in dataset")
        for cc in c.get("connected_card_ids", []):
            if cc not in valid_card:
                E(f"connected card {cc} not in dataset")
        for pc in c.get("similar_prior_cases", []):
            if pc not in valid_case:
                E(f"prior case {pc} not in closed_cases_history")
        for sub in s.get("subjects", []):
            if sub.startswith("C") and "-K" in sub and sub not in valid_card:
                W(f"SAR subject {sub} not a known card id")

        # ---- cross-field consistency ----------------------------------
        if c.get("pattern") == "undocumented" and not c.get("pattern_description"):
            E("pattern is 'undocumented' but pattern_description is empty")
        if c.get("pattern") != "undocumented" and c.get("pattern_description"):
            W("pattern_description set for a documented pattern")

        if c.get("verdict") == "legitimate":
            if c.get("affected_txn_ids"):
                E("legitimate verdict must have empty affected_txn_ids")
            if c.get("exposure_usd"):
                E("legitimate verdict must have exposure_usd = 0")
            if s.get("file"):
                E("legitimate verdict must not file a report")

        final = a.get("next_best_actions", {}).get("final", [])
        initial = a.get("next_best_actions", {}).get("initial", [])
        has_file = any(x.get("action") == "FILE_REPORT" for x in final)
        if bool(s.get("file")) != has_file:
            E(f"sar.file={s.get('file')} disagrees with FILE_REPORT in final actions")

        for label, lst in (("initial", initial), ("final", final)):
            if not lst:
                E(f"{label} actions are empty")
            for act in lst:
                if act.get("action") not in ACTIONS:
                    E(f"{label}: unknown action {act.get('action')!r}")
                if act.get("route") not in ROUTE:
                    E(f"{label}: bad route {act.get('route')!r}")
                if not act.get("reason"):
                    E(f"{label}: {act.get('action')} has no reason")
                # routing table
                an, rt = act.get("action"), act.get("route")
                if an in AUTO and rt != "auto":
                    E(f"{label}: {an} must route auto, got {rt}")
                if an == "DECLINE_TRANSACTION" and rt != "L1":
                    E(f"{label}: DECLINE_TRANSACTION must route L1")
                if an in ("BLOCK_ALL_CARDS", "FILE_REPORT") and rt != "L2":
                    E(f"{label}: {an} must route L2")
                if an == "BLOCK_CARD":
                    want = "L1" if (c.get("exposure_usd") or 0) <= 2500 else "L2"
                    if rt != want:
                        E(f"{label}: BLOCK_CARD at ${c.get('exposure_usd')} must route {want}")

        if a.get("evidence_requests"):
            for r in a["evidence_requests"]:
                if r.get("type") not in EVREQ:
                    E(f"bad evidence_request type {r.get('type')!r}")
                if not r.get("assumed_response"):
                    E("evidence_request has no assumed_response")
        else:
            if initial != final:
                E("no evidence requested but final differs from initial")

        # ---- SAR body -------------------------------------------------
        if s.get("file"):
            n = len(s.get("narrative", "").split("."))
            if n < 6:
                W(f"SAR narrative is short ({n} sentences; the format asks for 6-12)")
            if not s.get("subjects"):
                E("SAR filed with no subjects")
            if len(s.get("activity_dates", [])) != 2:
                E("SAR activity_dates must have exactly two dates")
            if not s.get("total_amount_usd"):
                W("SAR filed with total_amount_usd = 0")
        else:
            if s.get("narrative") or s.get("subjects") or s.get("total_amount_usd") \
               or s.get("activity_dates"):
                E("sar.file is false but the report body is not empty")

        # ---- evidence -------------------------------------------------
        if not c.get("evidence"):
            E("no evidence recorded")
        for ev in c.get("evidence", []):
            if ev.get("source") not in SOURCE:
                E(f"bad evidence source {ev.get('source')!r}")
            if not ev.get("claim") or not ev.get("ref"):
                E("evidence item missing claim or ref")
        if not c.get("summary"):
            E("empty summary")
        if not a.get("stop_reason"):
            E("empty stop_reason")

        # ---- exposure arithmetic --------------------------------------
        if c.get("affected_txn_ids"):
            tot = 0.0
            for t in c["affected_txn_ids"]:
                r = ds.txn_row(int(t))
                if r is not None:
                    tot += abs(float(r["TransactionAmt"]))
            if abs(tot - float(c.get("exposure_usd", 0))) > 0.02:
                E(f"exposure {c.get('exposure_usd')} != sum of affected amounts {tot:.2f}")

    missing = expected - seen
    if missing:
        errors.append(f"missing answer files for: {', '.join(sorted(missing))}")

    print(f"validated {len(files)} answer files")
    for w in warnings:
        print("  WARN  " + w)
    for e in errors:
        print("  FAIL  " + e)
    if errors:
        print(f"\n{len(errors)} error(s)")
        return 1
    print(f"\nall checks passed ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
