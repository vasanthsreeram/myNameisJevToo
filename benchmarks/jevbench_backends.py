#!/usr/bin/env python3
"""Run JevBench's public items against any backend: MLX weights or a llama.cpp server.

    python benchmarks/jevbench_backends.py mlx  /path/to/mlx-model
    python benchmarks/jevbench_backends.py gguf http://127.0.0.1:8080

Same rendering, same scoring formulas, so the two legs are comparable.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jevbench import DATASETS, TIER_CHANCES, TIER_WEIGHTS, ece_top_label, render  # noqa: E402

OUTDIR = Path("benchmarks/results")
OUTDIR.mkdir(parents=True, exist_ok=True)

TIERS = [("easy", "easy.jsonl"), ("standard", "original.jsonl"), ("hard", "hard.jsonl")]


def make_backend(kind, target):
    if kind == "mlx":
        from jevtoo.backends import MLXBackend
        return MLXBackend(target)
    if kind == "gguf":
        from jevtoo.backends_gguf import LlamaCppServerBackend
        return LlamaCppServerBackend(target)
    raise SystemExit(f"unknown backend {kind!r}")


def run_tier(backend, path, tier, blind=False, reverse_noul=False):
    recs = []
    errors = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        labels = list(r["labels"])
        q = r["question"]
        if reverse_noul and q.get("type") == "noul" and len(labels) == 2:
            labels = labels[::-1]
        state = "[state withheld]" if blind else r["state"]
        prompt = render(state, q, labels)
        try:
            d = backend.distribution(prompt, labels)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{r['id']}: {type(e).__name__}: {e}")
            continue
        best = max(range(len(labels)), key=lambda i: d.raw[i])
        correct = str(labels[best]) == str(r["expected"])
        ev = None
        if q.get("type") == "score":
            ev = sum(i * s for i, s in enumerate(d.share))
            correct = (round(ev) == r["expected"])
        recs.append({
            "id": r["id"], "tier": tier, "family": r["family"], "type": q.get("type"),
            "n_options": len(labels), "expected": r["expected"], "picked": labels[best],
            "correct": correct, "confidence": d.confidence,
            "share": d.share, "raw": d.raw, "ev": ev, "latency_s": d.latency_s,
        })
    if errors:
        print(f"  !! {len(errors)}/{len(errors)+len(recs)} items FAILED in {tier}")
        for e in errors[:3]:
            print(f"     {e}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["mlx", "gguf"])
    ap.add_argument("target")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-blind", action="store_true")
    args = ap.parse_args()

    tag = args.tag or args.kind
    backend = make_backend(args.kind, args.target)

    all_recs, t0 = {}, time.time()
    for tier, fname in TIERS:
        recs = run_tier(backend, f"{DATASETS}/{fname}", tier)
        if not recs:
            raise SystemExit(f"no items scored for tier {tier!r} - see errors above")
        all_recs[tier] = recs
        print(f"[{tag}/{tier}] {len(recs)} items in {time.time()-t0:.0f}s cumulative", flush=True)

    print("\n" + "=" * 74)
    print(f"{tag} - JevBench public items, letter-probability readout")
    print("=" * 74)
    num = den = 0.0
    print(f"{'tier':<10} {'n':>4} {'accuracy':>10} {'chance':>8} {'chance-corrected':>18}")
    for tier, _ in TIERS:
        recs = all_recs[tier]
        acc = sum(r["correct"] for r in recs) / len(recs)
        ch = TIER_CHANCES[tier]
        cc = max(0.0, min(100.0, 100 * (acc - ch) / (1 - ch)))
        num += TIER_WEIGHTS[tier] * cc
        den += TIER_WEIGHTS[tier]
        print(f"{tier:<10} {len(recs):>4} {acc*100:>9.1f}% {ch*100:>7.1f}% {cc:>17.1f}")
    intel = num / den
    print(f"{'WEIGHTED':<10} {'':>4} {'':>10} {'':>8} {intel:>17.1f}   <- intelligence axis")

    hard = all_recs["hard"]
    ece, _ = ece_top_label([(r["confidence"], r["correct"]) for r in hard])
    lat = sorted(r["latency_s"] for t, _ in TIERS for r in all_recs[t])
    p50, p95 = lat[int(.5 * (len(lat) - 1))], lat[int(.95 * (len(lat) - 1))]
    adj50, adj95 = p50 * 2 + .15, p95 * 2 + .15
    sp = (max(0, min(100, 100 - 20 * math.log10(adj50 / .1))) +
          max(0, min(100, 100 - 20 * math.log10(adj95 / .1)))) / 2
    cal = max(0.0, 100 * (1 - ece / 0.5))
    g3 = math.exp(sum(math.log(max(x, 1e-9)) for x in (intel, cal, sp)) / 3)
    mult = 1.0 if intel >= 50 else (max(intel, 0) / 50) ** 2
    print(f"\nhard-tier ECE = {ece:.4f}   calibration axis = {cal:.1f}")
    print(f"latency raw p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms | "
          f"adjusted p50={adj50:.2f}s p95={adj95:.2f}s  speed axis = {sp:.1f}")
    print(f"axes: intelligence={intel:.1f} calibration={cal:.1f} speed={sp:.1f}")
    print(f"score equivalent (3 axes, cost excluded) = {g3*mult:.1f}")

    blind = {}
    if not args.skip_blind:
        print("\nstate-blind control:")
        for tier, fname in TIERS:
            recs = [r for r in run_tier(backend, f"{DATASETS}/{fname}", tier, blind=True)
                    if "error" not in r]
            acc = sum(r["correct"] for r in recs) / len(recs)
            with_state = sum(r["correct"] for r in all_recs[tier]) / len(all_recs[tier])
            blind[tier] = acc
            print(f"  {tier:<9} with={with_state*100:5.1f}%  without={acc*100:5.1f}%  "
                  f"kept={acc/with_state*100:4.0f}%")
        all_recs["stateblind"] = blind

    cov = getattr(backend, "coverage", None)
    summary = {"tag": tag, "intelligence": intel, "hard_ece": ece, "calibration": cal,
               "speed": sp, "p50_s": p50, "p95_s": p95, "score_equivalent_3axes": g3 * mult,
               "stateblind": blind,
               "label_coverage": cov() if callable(cov) else None,
               "stats": getattr(backend, "stats", None)}
    json.dump(summary, open(OUTDIR / f"jevbench_{tag}.json", "w"), indent=1)
    json.dump({k: v for k, v in all_recs.items()}, open(OUTDIR / f"jevbench_{tag}_detail.json", "w"), indent=1)
    print(f"\n-> benchmarks/results/jevbench_{tag}.json")


if __name__ == "__main__":
    main()
