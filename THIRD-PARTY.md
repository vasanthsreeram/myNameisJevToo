# Third-party material and attribution

This repository contains no model weights and no third-party task text. The notes below cover
the names, data and numbers that are referenced rather than redistributed.

## TypeSafe AI — "Jev", "System One"

"Jev" and "System One" are TypeSafe AI's names for their model and model class.

**This project is not affiliated with, endorsed by, or connected to TypeSafe AI.** The name
`myNameisJevToo` is an affectionate nod to them, and every Jev figure quoted in this
repository's documentation is TypeSafe's own published number, cited to where they published it.
No TypeSafe code, weights or data are included here. `POST /v1/systemone` follows the
request and answer fields in their published API reference. The hero image is the
"my name is Jeff" still with a wordmark drawn for this repository (the pink is the
`#E551BA` used on their confidence page). It is not a TypeSafe logo file.

## Model weights — OpenBMB / MiniCPM5-2B

MiniCPM5-2B and MiniCPM5-2B-MLX are OpenBMB's, released under Apache-2.0. They are **downloaded,
never vendored** — this repository contains no model weights. The Qwen3.8-27B checkpoints
referenced in the results are likewise downloaded, not redistributed.

## Benchmark data — fstandhartinger/jevbench

JevBench task items and scoring formulas come from
[`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench). Its public items are
**fetched on demand** by `datasets/fetch_jevbench.sh` and are never committed to this repository —
the task text remains under that project's own terms. Check those terms before redistributing the
task text.

What *is* committed under `benchmarks/results/` is our own output: per-item identifiers, the
probability we read for each option, which option was picked, and our latencies. That is our
measurement record, published so the numbers in the documentation can be audited rather than
taken on trust.

## Other referenced work

- [`jujumilk3/jev-calibration-audit`](https://github.com/jujumilk3/jev-calibration-audit) — the
  state-blind and abstain-option findings that shaped the controls used here.
- `convaiinnovations/laya`, `logan-markewich/jeff`, `Fastino/GLiNER2` — open decision models
  referenced as comparison points on the same board.

Full credit list: see the **Credits** section of [README.md](README.md).
