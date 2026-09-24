# Autonomous monitoring — 2016-11-01 to 2016-12-31

The agent scanned the exam period on its own, without an alert, for rare device profiles carrying many unrelated cardholders.

- **362 rings** found
- **1330 distinct cards** across them
- **$531,647.49** of transactions
- **1** of the 362 rings have a median model score above 0.7

The last line is the finding. These clusters are almost invisible to a transaction-level score: every individual payment looks ordinary, and the pattern only exists as a shape in the graph.

| cards | lifetime | conc | median risk | amount | product | device profile |
|---|---|---|---|---|---|---|
| 55 | 55 | 1.00 | 0.22 | $6,290 | S×36, R×23 | `Windows | Windows 10 | chrome 66.0 | 1600x900` |
| 49 | 50 | 0.98 | 0.29 | $10,326 | H×48, R×36 | `iOS Device | iOS 11.3.0 | mobile safari 11.0 | 1136x640` |
| 48 | 48 | 1.00 | 0.39 | $8,639 | R×40, H×31 | `iOS Device | iOS 11.3.0 | mobile safari generic | 2048x153` |
| 47 | 48 | 0.98 | 0.17 | $8,150 | R×32, H×19 | `Windows | Windows 7 | chrome 66.0 | 1366x768` |
| 45 | 47 | 0.96 | 0.15 | $6,085 | R×32, H×20 | `Windows | Windows 7 | chrome 66.0 | 1600x900` |
| 44 | 56 | 0.79 | 0.15 | $5,310 | R×31, H×9 | `Windows | Windows 7 | chrome 65.0 | 1600x900` |
| 42 | 43 | 0.98 | 0.37 | $4,637 | R×31, H×20 | `iOS Device | iOS 11.3.0 | mobile safari generic | 1136x640` |
| 42 | 58 | 0.72 | 0.16 | $4,279 | H×29, S×18 | `Windows | Windows 10 | chrome generic | 1366x768` |
| 40 | 41 | 0.98 | 0.40 | $5,129 | H×33, R×21 | `iOS Device | iOS 11.3.0 | mobile safari generic | 2001x112` |
| 37 | 39 | 0.95 | 0.39 | $5,261 | R×28, H×20 | `iOS Device | iOS 11.3.0 | mobile safari 11.0 | 2001x1125` |
| 36 | 51 | 0.71 | 0.12 | $3,284 | C×74 | `rv:59.0 | unknown-os | firefox 59.0 | unknown-screen` |
| 36 | 39 | 0.92 | 0.15 | $2,715 | C×57, R×1 | `Windows | unknown-os | firefox 60.0 | unknown-screen` |
| 35 | 44 | 0.80 | 0.16 | $6,150 | R×25, H×17 | `Windows | Windows 7 | chrome 65.0 | 1280x1024` |
| 35 | 36 | 0.97 | 0.49 | $3,905 | R×22, H×20 | `iOS Device | iOS 11.2.6 | mobile safari 11.0 | 1334x750` |
| 32 | 34 | 0.94 | 0.28 | $4,740 | R×18, H×13 | `MacOS | Mac OS X 10_13_4 | safari 11.0 | 2880x1800` |
| 32 | 32 | 1.00 | 0.15 | $4,476 | R×22, H×17 | `MacOS | Mac OS X 10_13_4 | safari 11.0 | 1920x1080` |
| 30 | 34 | 0.88 | 0.19 | $7,259 | R×35, H×9 | `Windows | Windows 7 | firefox 59.0 | 1920x1080` |
| 29 | 32 | 0.91 | 0.10 | $25,532 | S×229, R×15 | `Windows | Windows 10 | edge 16.0 | 1366x767` |
| 28 | 52 | 0.54 | 0.13 | $8,869 | C×60 | `SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for andr` |
| 28 | 29 | 0.97 | 0.15 | $4,860 | R×24, H×10 | `Windows | Windows 7 | chrome 66.0 | 1680x1050` |
| 27 | 27 | 1.00 | 0.26 | $3,570 | R×24, H×13 | `MacOS | Mac OS X 10_13_4 | safari generic | 1440x900` |
| 27 | 29 | 0.93 | 0.13 | $2,347 | R×15, S×11 | `Windows | Windows 10 | chrome 65.0 | 1280x1024` |
| 26 | 28 | 0.93 | 0.21 | $6,536 | R×25, H×18 | `MacOS | Mac OS X 10_13_4 | safari 11.0 | 1440x900` |
| 26 | 38 | 0.68 | 0.32 | $3,190 | R×15, H×6 | `Windows | Windows 8.1 | chrome 65.0 | 1920x1080` |
| 26 | 32 | 0.81 | 0.14 | $2,555 | C×77 | `SM-J700M Build/MMB29K | unknown-os | chrome 65.0 for andro` |

Full detail, including every card id, is in `ring_findings.json`.

Recommended handling under the policy: `CREATE_CASE`, `MONITOR_CONNECTED_CARDS` for every card on the profile, and `ESCALATE_TO_ANALYST` under R9 (undocumented, coordinated across customers). Filing is R9 + 3a where exposure or cross-customer linkage qualifies.
