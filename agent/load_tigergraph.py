"""
One-command TigerGraph setup.

    python -m agent.load_tigergraph --schema     # create graph + schema
    python -m agent.load_tigergraph --queries    # install the GSQL queries
    python -m agent.load_tigergraph --load       # upsert the CSVs over REST++
    python -m agent.load_tigergraph --all

The loader batches REST++ upserts rather than using a server-side loading job,
so it works against Savanna without needing the CSVs on the database host.
For a local Community Edition install, `gsql/02_load.gsql` is faster: put the
CSVs where the server can see them and RUN LOADING JOB load_fraud_graph.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tg_client import from_env, TigerGraphError   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHDIR = os.path.join(ROOT, "data", "graph")
GSQLDIR = os.path.join(ROOT, "gsql")

BATCH = 2000

VERTEX_FILES = [
    ("customer.csv", "Customer", "customer_id",
     ["n_cards", "n_txns", "first_seen", "last_seen", "home_region", "total_spend"]),
    ("card.csv", "Card", "card_id",
     ["phys_card", "customer_id", "card1", "network", "card_type", "n_txns",
      "first_seen", "last_seen", "amount_median", "amount_p95"]),
    ("device_profile.csv", "DeviceProfile", "device_key",
     ["device_info", "os", "browser", "screen", "device_type", "n_txns",
      "n_cards_total", "is_specific"]),
    ("email_domain.csv", "EmailDomain", "domain", ["n_txns"]),
    ("billing_region.csv", "BillingRegion", "region_code", ["country", "n_txns", "n_cards"]),
    ("closed_case.csv", "ClosedCase", "case_id",
     ["customer_id", "card_id", "opened_at", "closed_at", "outcome", "pattern",
      "n_txns", "exposure_usd", "actions_taken", "report_filed", "analyst_notes"]),
    ("transaction.csv", "Transaction", "txn_id",
     ["ts", "amount", "product_cd", "channel", "risk_score", "addr1", "addr2",
      "dist1", "p_email", "r_email", "device_status", "proxy_status",
      "match_false_count", "customer_id", "card_id"]),
]

EDGE_FILES = [
    ("e_owns.csv", "Customer", "customer_id", "OWNS", "Card", "card_id"),
    ("e_made.csv", "Card", "card_id", "MADE", "Transaction", "txn_id"),
    ("e_from_device.csv", "Transaction", "txn_id", "FROM_DEVICE", "DeviceProfile", "device_key"),
    ("e_purchaser_email.csv", "Transaction", "txn_id", "PURCHASER_EMAIL", "EmailDomain", "domain"),
    ("e_billed_in.csv", "Transaction", "txn_id", "BILLED_IN", "BillingRegion", "region_code"),
    ("e_next.csv", "Transaction", "a", "NEXT", "Transaction", "b"),
    ("e_involves.csv", "ClosedCase", "case_id", "INVOLVES", "Transaction", "txn_id"),
    ("e_on_card.csv", "ClosedCase", "case_id", "ON_CARD", "Card", "card_id"),
    ("e_connected_to.csv", "ClosedCase", "case_id", "CONNECTED_TO", "Card", "card_id"),
]


def run_schema(c, prefix: str = ""):
    """
    Create the global types, then bind them into the graph.

    `prefix` namespaces every type name.  Global type names are shared across a
    whole workspace, so if another graph there already defines Card or
    Transaction, creation fails; --prefix FI_ sidesteps it without touching
    anyone else's schema.
    """
    with open(os.path.join(GSQLDIR, "01_schema.gsql")) as fh:
        ddl = fh.read()
    if prefix:
        import re
        names = re.findall(r"CREATE (?:DIRECTED EDGE|VERTEX)\s+(\w+)", ddl)
        names += re.findall(r'REVERSE_EDGE="(\w+)"', ddl)
        for n in sorted(set(names), key=len, reverse=True):
            ddl = re.sub(rf"\b{n}\b", prefix + n, ddl)
        ddl = ddl.replace(prefix + "Fraud_Investigation", "Fraud_Investigation")
    print(c.gsql(ddl)[-1500:])


def run_queries(c):
    with open(os.path.join(GSQLDIR, "03_queries.gsql")) as fh:
        print(c.gsql(fh.read())[-1500:])
    print(c.gsql(f"USE GRAPH {c.graph}\nINSTALL QUERY ALL")[-800:])


def _rows(path):
    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)


def load_vertices(c, only=None):
    for fname, vtype, key, attrs in VERTEX_FILES:
        if only and vtype not in only:
            continue
        path = os.path.join(GRAPHDIR, fname)
        if not os.path.exists(path):
            print(f"  skip {fname} (not built)")
            continue
        t0, n, batch = time.time(), 0, {}
        for row in _rows(path):
            vid = row[key]
            if not vid:
                continue
            batch[vid] = {a: {"value": row.get(a, "")} for a in attrs}
            if len(batch) >= BATCH:
                c.upsert(vertices={vtype: batch}); n += len(batch); batch = {}
        if batch:
            c.upsert(vertices={vtype: batch}); n += len(batch)
        print(f"  {vtype:<15} {n:>8,} vertices  ({time.time()-t0:.0f}s)")


def load_edges(c):
    for fname, sv, skey, etype, tv, tkey in EDGE_FILES:
        path = os.path.join(GRAPHDIR, fname)
        if not os.path.exists(path):
            print(f"  skip {fname}")
            continue
        t0, n, batch = time.time(), 0, {}
        for row in _rows(path):
            s, t = row.get(skey), row.get(tkey)
            if not s or not t or t == "-1":
                continue
            batch.setdefault(s, {etype: {tv: {}}})[etype][tv][t] = {}
            if sum(len(v[etype][tv]) for v in batch.values()) >= BATCH:
                c.upsert(edges={sv: batch}); n += len(batch); batch = {}
        if batch:
            c.upsert(edges={sv: batch}); n += len(batch)
        print(f"  {etype:<18} {n:>8,} source vertices ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prefix", default="",
                    help="namespace all type names, e.g. FI_ (use when the "
                         "workspace already defines Card/Transaction globally)")
    a = ap.parse_args()

    c = from_env()
    if not c.ping():
        raise TigerGraphError("workspace is not reachable/running")
    print(f"connected: {c.host}  graph={c.graph}")

    if a.schema or a.all:
        print("creating schema...");  run_schema(c, a.prefix)
    if a.queries or a.all:
        print("installing queries..."); run_queries(c)
    if a.load or a.all:
        print("loading vertices..."); load_vertices(c)
        print("loading edges...");    load_edges(c)
    print("done")


if __name__ == "__main__":
    main()
