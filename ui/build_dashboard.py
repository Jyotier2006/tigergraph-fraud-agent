"""
Build the analyst dashboard.

    python ui/build_dashboard.py     ->  ui/dashboard.html

The dashboard is a single self-contained file: it embeds the 20 answer files
and the agent's reasoning trace, so it opens from disk with no server and no
network.  It is the investigation view an analyst would work in -- queue,
case record, evidence with its graph references, the uncertainty step, and the
before/after next-best-action.
"""
from __future__ import annotations

import json
import os
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "cases")
OUT = os.path.join(ROOT, "ui", "dashboard.html")

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Investigation Console</title>
<style>
  :root {
    --bg:#0f1115; --panel:#161a21; --panel2:#1c212a; --line:#272e3a;
    --ink:#e6e9ef; --muted:#93a0b4; --accent:#e8873a; --accent2:#4da3ff;
    --fraud:#e5534b; --legit:#3fb950; --uncertain:#d29922;
    --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0; letter-spacing:.2px; }
  header .sub { color:var(--muted); font-size:13px; }
  .stats { margin-left:auto; display:flex; gap:20px; }
  .stat b { display:block; font-size:19px; }
  .stat span { color:var(--muted); font-size:11px; text-transform:uppercase;
               letter-spacing:.6px; }
  .wrap { display:grid; grid-template-columns:310px 1fr; min-height:calc(100vh - 66px); }
  .queue { border-right:1px solid var(--line); overflow-y:auto; max-height:calc(100vh - 66px); }
  .q { padding:11px 16px; border-bottom:1px solid var(--line); cursor:pointer; }
  .q:hover { background:var(--panel); }
  .q.on { background:var(--panel2); border-left:3px solid var(--accent); padding-left:13px; }
  .q .id { font-family:var(--mono); font-size:12px; }
  .q .meta { color:var(--muted); font-size:11.5px; margin-top:3px; }
  .pill { display:inline-block; padding:1px 7px; border-radius:99px; font-size:10.5px;
          font-weight:600; letter-spacing:.3px; }
  .fraud{background:rgba(229,83,75,.16); color:var(--fraud);}
  .legitimate{background:rgba(63,185,80,.14); color:var(--legit);}
  .uncertain{background:rgba(210,153,34,.16); color:var(--uncertain);}
  main { padding:22px 26px; overflow-y:auto; max-height:calc(100vh - 66px); }
  h2 { font-size:15px; margin:0 0 4px; }
  .crumb { color:var(--muted); font-size:12.5px; margin-bottom:16px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px;
            padding:15px 17px; margin-bottom:14px; }
  section h3 { font-size:11px; text-transform:uppercase; letter-spacing:.7px;
               color:var(--muted); margin:0 0 10px; font-weight:600; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:12px; }
  .kv b { display:block; font-size:15px; }
  .kv span { color:var(--muted); font-size:11px; }
  .bar { height:5px; background:var(--panel2); border-radius:3px; overflow:hidden; margin-top:7px;}
  .bar i { display:block; height:100%; background:var(--accent); }
  .ev { border-left:2px solid var(--line); padding:0 0 0 12px; margin-bottom:13px; }
  .ev p { margin:0 0 5px; }
  .ev .ref { font-family:var(--mono); font-size:11px; color:var(--accent2);
             word-break:break-all; }
  .ev .src { font-size:10.5px; color:var(--muted); text-transform:uppercase;
             letter-spacing:.5px; }
  .act { display:flex; gap:10px; align-items:flex-start; padding:8px 0;
         border-bottom:1px solid var(--line); }
  .act:last-child { border-bottom:0; }
  .act code { font-family:var(--mono); font-size:12px; color:var(--ink); min-width:210px; }
  .route { font-size:10px; padding:1px 6px; border-radius:4px; font-weight:700; }
  .auto{background:rgba(63,185,80,.14); color:var(--legit);}
  .L1{background:rgba(210,153,34,.16); color:var(--uncertain);}
  .L2{background:rgba(229,83,75,.16); color:var(--fraud);}
  .act .why { color:var(--muted); font-size:12.5px; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:900px){ .wrap{grid-template-columns:1fr} .two{grid-template-columns:1fr} }
  .chips span { display:inline-block; font-family:var(--mono); font-size:11px;
                background:var(--panel2); border:1px solid var(--line);
                padding:2px 7px; border-radius:5px; margin:0 5px 5px 0; }
  .narr { white-space:pre-wrap; background:var(--panel2); padding:12px;
          border-radius:6px; font-size:13px; }
  .steps li { margin-bottom:3px; font-family:var(--mono); font-size:11.5px;
              color:var(--muted); }
  .tag { font-family:var(--mono); font-size:11px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Fraud Investigation Console</h1>
  <span class="sub">TigerGraph &middot; agentic case review &middot; HHGOA case pack</span>
  <div class="stats" id="stats"></div>
</header>
<div class="wrap">
  <div class="queue" id="queue"></div>
  <main id="detail"></main>
</div>
<script>
const CASES = __DATA__;

const money = n => '$' + (n||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const esc = s => String(s==null?'':s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function stats(){
  const v = {fraud:0, legitimate:0, uncertain:0};
  let sar=0, exp=0;
  CASES.forEach(c=>{ v[c.case.verdict]++; if(c.sar.file) sar++; exp += c.case.exposure_usd; });
  document.getElementById('stats').innerHTML = `
    <div class="stat"><b>${CASES.length}</b><span>cases</span></div>
    <div class="stat"><b style="color:var(--fraud)">${v.fraud}</b><span>fraud</span></div>
    <div class="stat"><b style="color:var(--legit)">${v.legitimate}</b><span>legitimate</span></div>
    <div class="stat"><b style="color:var(--uncertain)">${v.uncertain}</b><span>uncertain</span></div>
    <div class="stat"><b>${sar}</b><span>reports</span></div>
    <div class="stat"><b>${money(exp)}</b><span>exposure</span></div>`;
}

function queue(sel){
  document.getElementById('queue').innerHTML = CASES.map(c=>`
    <div class="q ${c.case_id===sel?'on':''}" onclick="show('${c.case_id}')">
      <div class="id">${c.case_id} <span class="pill ${c.case.verdict}">${c.case.verdict}</span></div>
      <div class="meta">p=${c.case.fraud_probability.toFixed(2)} &middot; ${c.case.pattern.replace(/_/g,' ')}
        ${c.case.exposure_usd? '&middot; '+money(c.case.exposure_usd):''}</div>
    </div>`).join('');
}

function actions(list){
  return list.map(a=>`<div class="act"><code>${a.action}</code>
    <span class="route ${a.route}">${a.route}</span>
    <span class="why">${esc(a.reason)}</span></div>`).join('');
}

function show(id){
  const c = CASES.find(x=>x.case_id===id); if(!c) return;
  queue(id);
  const k = c.case;
  document.getElementById('detail').innerHTML = `
    <h2>${c.case_id} <span class="pill ${k.verdict}">${k.verdict}</span></h2>
    <div class="crumb">${esc(k.status)} &middot; ${esc(k.pattern.replace(/_/g,' '))}
      &middot; ${c.tool_calls} graph calls &middot; ${c.latency_s}s</div>

    <section>
      <div class="grid">
        <div class="kv"><b>${k.fraud_probability.toFixed(2)}</b><span>fraud probability</span>
          <div class="bar"><i style="width:${k.fraud_probability*100}%"></i></div></div>
        <div class="kv"><b>${money(k.exposure_usd)}</b><span>exposure</span></div>
        <div class="kv"><b>${k.affected_txn_ids.length}</b><span>affected transactions</span></div>
        <div class="kv"><b>${k.connected_card_ids.length}</b><span>connected cards</span></div>
        <div class="kv"><b>${k.similar_prior_cases.length}</b><span>prior cases used</span></div>
        <div class="kv"><b>${c.sar.file?'filed':'none'}</b><span>regulatory report</span></div>
      </div>
    </section>

    <section><h3>Summary</h3><p>${esc(k.summary)}</p>
      ${k.pattern_description?`<p style="color:var(--muted)">${esc(k.pattern_description)}</p>`:''}
    </section>

    <section><h3>Evidence &middot; every claim carries the graph call behind it</h3>
      ${k.evidence.map(e=>`<div class="ev">
         <p>${esc(e.claim)}</p>
         <div class="src">${e.source}</div>
         <div class="ref">${esc(e.ref)}</div>
       </div>`).join('')}
    </section>

    ${c.evidence_requests.length?`<section><h3>Uncertainty &middot; evidence requested before acting</h3>
      ${c.evidence_requests.map(r=>`<p><b>${r.type.replace(/_/g,' ')}</b> (after step ${r.asked_after_step})<br>
        <span style="color:var(--muted)">${esc(r.assumed_response)}</span></p>`).join('')}
    </section>`:''}

    <section><h3>Next best action</h3>
      <div class="two">
        <div><div class="tag">INITIAL &mdash; before requested evidence</div>${actions(c.next_best_actions.initial)}</div>
        <div><div class="tag">FINAL &mdash; after the response</div>${actions(c.next_best_actions.final)}</div>
      </div>
      <p style="color:var(--muted);margin-top:10px">${esc(c.next_best_actions.what_changed)}</p>
    </section>

    ${c.sar.file?`<section><h3>Suspicious activity report</h3>
      <p style="color:var(--muted)">${esc(c.sar.reason)}</p>
      <div class="narr">${esc(c.sar.narrative)}</div>
      <p style="margin-top:10px"><span class="tag">SUBJECTS</span></p>
      <div class="chips">${c.sar.subjects.map(s=>`<span>${esc(s)}</span>`).join('')}</div>
      <p class="tag">${c.sar.activity_dates.join(' to ')} &middot; ${money(c.sar.total_amount_usd)}</p>
    </section>`:`<section><h3>Suspicious activity report</h3>
      <p style="color:var(--muted)">Not filed. ${esc(c.sar.reason)}</p></section>`}

    ${k.connected_card_ids.length?`<section><h3>Connected cards &middot; shared device profile</h3>
      <div class="chips">${k.connected_card_ids.map(x=>`<span>${esc(x)}</span>`).join('')}</div>
      ${k.connected_device_profiles.map(d=>`<p class="ref" style="font-family:var(--mono);font-size:11.5px;color:var(--accent2)">${esc(d)}</p>`).join('')}
    </section>`:''}

    <section><h3>Stop decision</h3><p>${esc(c.stop_reason)}</p>
      <p class="tag">affected: ${k.affected_txn_ids.join(', ')||'none'}</p>
      <p class="tag">prior cases: ${k.similar_prior_cases.join(', ')||'none'}</p>
      <p class="tag">written to graph: ${k.written_to_graph} ${k.graph_case_id?('('+k.graph_case_id+')'):''}</p>
    </section>`;
}

stats(); queue(CASES[0].case_id); show(CASES[0].case_id);
</script>
</body>
</html>
"""


def main():
    files = sorted(glob.glob(os.path.join(CASES, "HHG-*.json")))
    data = [json.load(open(f)) for f in files]
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT}  ({len(html)/1024:.0f} KB, {len(data)} cases)")


if __name__ == "__main__":
    main()
