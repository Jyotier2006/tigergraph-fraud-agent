"""
TigerGraph connection: REST++ / GSQL over HTTPS.

Configuration comes from the environment (see .env.example):

    TG_HOST      https://<workspace>.i.tgcloud.io
    TG_SECRET    a database secret created in Savanna -> Database Secrets
    TG_GRAPH     Fraud_Investigation

The client is deliberately thin -- requests + a token -- so it runs anywhere
pyTigerGraph does, and so the MCP server, the loader and the agent all share
one code path.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

DEFAULT_GRAPH = os.environ.get("TG_GRAPH", "Fraud_Investigation")


class TigerGraphError(RuntimeError):
    pass


class TigerGraphClient:
    def __init__(self, host: str | None = None, secret: str | None = None,
                 graph: str | None = None, timeout: int = 120):
        self.host = (host or os.environ.get("TG_HOST", "")).rstrip("/")
        self.secret = secret or os.environ.get("TG_SECRET", "")
        self.graph = graph or DEFAULT_GRAPH
        self.timeout = timeout
        self._token: str | None = None
        if not self.host:
            raise TigerGraphError("TG_HOST is not set")

    # ------------------------------------------------------------- auth
    @property
    def token(self) -> str:
        if self._token:
            return self._token
        # TigerGraph 4.x exposes /gsql/v1/tokens; 3.x used /restpp/requesttoken.
        for path, payload in (
            ("/gsql/v1/tokens", {"secret": self.secret, "lifetime": 2592000}),
            ("/restpp/requesttoken", {"secret": self.secret, "lifetime": "2592000"}),
        ):
            try:
                r = requests.post(self.host + path, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    j = r.json()
                    tok = j.get("token") or (j.get("results") or {}).get("token")
                    if tok:
                        self._token = tok
                        return tok
            except Exception:
                continue
        raise TigerGraphError("could not obtain a token; check TG_SECRET and that the "
                              "workspace is running")

    @property
    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    # ------------------------------------------------------------- gsql
    @property
    def _gsql_auth(self):
        """GSQL statements authenticate with the database secret over Basic auth."""
        return ("__GSQL__secret", self.secret)

    def gsql(self, statement: str) -> str:
        """Run GSQL. Used for schema creation, loading jobs and INSTALL QUERY."""
        last = ""
        for path in ("/gsql/v1/statements", "/gsqlserver/gsql/file"):
            try:
                r = requests.post(self.host + path, data=statement.encode(),
                                  auth=self._gsql_auth,
                                  headers={"Content-Type": "text/plain"},
                                  timeout=self.timeout)
                if r.status_code < 400:
                    return r.text
                last = f"{r.status_code}: {r.text[:200]}"
            except Exception as exc:
                last = str(exc)
        raise TigerGraphError(f"GSQL endpoint unavailable ({last})")

    # ------------------------------------------------------------- data
    def upsert(self, vertices: dict | None = None, edges: dict | None = None) -> dict:
        """POST /restpp/graph/{graph} -- the batch upsert used by the loader."""
        body: dict[str, Any] = {}
        if vertices:
            body["vertices"] = vertices
        if edges:
            body["edges"] = edges
        r = requests.post(f"{self.host}/restpp/graph/{self.graph}",
                          data=json.dumps(body), headers=self._h, timeout=self.timeout)
        if r.status_code >= 400:
            raise TigerGraphError(f"upsert failed {r.status_code}: {r.text[:300]}")
        return r.json()

    def run_query(self, name: str, params: dict | None = None) -> Any:
        r = requests.get(f"{self.host}/restpp/query/{self.graph}/{name}",
                         params=params or {}, headers=self._h, timeout=self.timeout)
        if r.status_code >= 400:
            raise TigerGraphError(f"query {name} failed {r.status_code}: {r.text[:300]}")
        j = r.json()
        if j.get("error"):
            raise TigerGraphError(f"query {name}: {j.get('message')}")
        return j.get("results", [])

    def vertices(self, vtype: str, vid: str | None = None, **kw) -> Any:
        url = f"{self.host}/restpp/graph/{self.graph}/vertices/{vtype}"
        if vid is not None:
            url += "/" + requests.utils.quote(str(vid), safe="")
        r = requests.get(url, headers=self._h, params=kw, timeout=self.timeout)
        if r.status_code >= 400:
            raise TigerGraphError(f"vertex fetch failed {r.status_code}: {r.text[:200]}")
        return r.json().get("results", [])

    def ping(self) -> bool:
        try:
            r = requests.get(self.host + "/restpp/echo", timeout=20)
            return r.status_code == 200 and "Hello GSQL" in r.text
        except Exception:
            return False


def from_env() -> TigerGraphClient:
    return TigerGraphClient()
