#!/usr/bin/env python3
"""Rebuild a run summary from its per-item detail file.

The summary and the detail file are written in the same pass, so a run that
dies between the two - or a summary that predates a field being persisted -
leaves numbers that cannot be audited. The detail file holds every per-item
record, so the summary is always reconstructible from it.

    python benchmarks/rescore.py benchmarks/results/jevbench_<tag>_detail.json

Uses the same formulas as jevbench_backends.py, imported from the same module,
so a rescored summary is identical to a freshly written one.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jevbench import TIER_CHANCES, TIER_WEIGHTS, ece_top_label  # noqa: E402

TIERS = ["easy", "standard", "hard"]


def rescore(detail_path, out_path=None):
    detail = json.load(open(detail_path))
    tag = Path(detail_path).name.replace("jevbench_", "").replace("_detail.json", "")

    all_recs = {t: detail[t] for t in TIERS if t in detail}
    missing = [t for t in TIERS if t not in all_recs]
    if missing:
        raise SystemExit(f"detail file is missing tiers: {missing}")

    tier_stats, num, den = {}, 0.0, 0.0
    for tier in TIERS:
        recs = all_recs[tier]
        acc = sum(r["correct"] for r in recs) / len(recs)
        ch = TIER_CHANCES[tier]
        cc = max(0.0, min(100.0, 100 * (acc - ch) / (1 - ch)))
        tier_stats[tier] = {"n": len(recs), "accuracy": acc, "chance": ch,
                            "chance_corrected": cc}
        num += TIER_WEIGHTS[tier] * cc
        den += TIER_WEIGHTS[tier]
    intel = num / den

    hard = all_recs["hard"]
    ece, _ = ece_top_label([(r["confidence"], r["correct"]) for r in hard])
    lat = sorted(r["latency_s"] for t in TIERS for r in all_recs[t])
    p50, p95 = lat[int(.5 * (len(lat) - 1))], lat[int(.95 * (len(lat) - 1))]
    adj50, adj95 = p50 * 2 + .15, p95 * 2 + .15
    sp = (max(0, min(100, 100 - 20 * math.log10(adj50 / .1))) +
          max(0, min(100, 100 - 20 * math.log10(adj95 / .1)))) / 2
    cal = max(0.0, 100 * (1 - ece / 0.5))
    g3 = math.exp(sum(math.log(max(x, 1e-9)) for x in (intel, cal, sp)) / 3)
    mult = 1.0 if intel >= 50 else (max(intel, 0) / 50) ** 2

    blind = detail.get("stateblind") or {}
    kept = {}
    for tier in TIERS:
        if tier in blind and all_recs[tier]:
            with_state = sum(r["correct"] for r in all_recs[tier]) / len(all_recs[tier])
            kept[tier] = (blind[tier] / with_state * 100) if with_state else None

    summary = {
        "tag": tag,
        "intelligence": intel, "hard_ece": ece, "calibration": cal, "speed": sp,
        "p50_s": p50, "p95_s": p95,
        "adjusted_p50_s": adj50, "adjusted_p95_s": adj95,
        "score_equivalent_3axes": g3 * mult,
        "tiers": tier_stats,
        "tier_weights": {t: TIER_WEIGHTS[t] for t in TIERS},
        "stateblind": blind, "stateblind_kept_pct": kept,
        # Item records in this file, i.e. the main pass only. Deliberately NOT
        # named "decisions": the backend's own counter also counts the
        # state-blind pass, so the same word would mean two different numbers.
        "item_records": sum(len(all_recs[t]) for t in TIERS),
        "_source": "rescored from detail file via benchmarks/rescore.py",
    }

    out_path = out_path or str(Path(detail_path).with_name(
        Path(detail_path).name.replace("_detail.json", ".json")))
    json.dump(summary, open(out_path, "w"), indent=1)

    print(f"rescored {tag}")
    for t in TIERS:
        s = tier_stats[t]
        print(f"  {t:<9} n={s['n']:<4} acc={s['accuracy']*100:5.1f}%  cc={s['chance_corrected']:5.1f}")
    print(f"  intelligence={intel:.1f}  calibration={cal:.1f}  speed={sp:.1f}")
    print(f"  raw p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms | "
          f"adjusted p50={adj50:.2f}s p95={adj95:.2f}s")
    print(f"  score equivalent (3 axes) = {g3*mult:.1f}")
    print(f"  -> {out_path}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rescore(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
