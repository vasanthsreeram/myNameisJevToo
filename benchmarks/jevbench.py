#!/usr/bin/env python3
"""
jevbench_minicpm.py - run MiniCPM5-2B (4-bit MLX) on JevBench's frozen public items.

Not a text generator: one forward pass per decision, read the probability of each
option's letter token off softmax(logits[-1]), exactly as specified.

Tiers present in the public halves:
  easy     -> datasets/public/easy.jsonl      (48 items)
  standard -> datasets/public/original.jsonl  (72 items)
  hard     -> datasets/public/hard.jsonl      (111 items)
(router / judge / held-out splits are not public, so the judge tier is not runnable.)

Scored with JevBench's own published formulas (jevbench/composite_v13.py,
jevbench/metrics.py): chance-corrected intelligence using their published
per-tier option-count baselines, ECE top-label 10 bins, speed, cost.
"""
import json
import math
import sys
import time

import mlx.core as mx
from mlx_lm import load

from _shared_ import _encode  # noqa: E402

MODEL = __import__("os").environ.get("JEV_MODEL", "openbmb/MiniCPM5-2B-MLX")
DATASETS = __import__("os").environ.get("JEVBENCH_DATA", "datasets/jevbench")
OUT = "benchmarks/results/jevbench_results.json"

# JevBench published tier baselines (from their option-count histograms)
TIER_OPTION_COUNTS = {
    "easy": {2: 18, 4: 13, 5: 41},
    "standard": {2: 32, 4: 40, 5: 12, 6: 12},
    "judge": {2: 68, 9: 78},
    "hard": {2: 77, 3: 26, 4: 73, 5: 38, 6: 6},
}
TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}

LETTERS = [chr(65 + i) for i in range(12)]


def chance_of(counts):
    n = sum(counts.values())
    return sum(c / o for o, c in counts.items()) / n


TIER_CHANCES = {t: chance_of(c) for t, c in TIER_OPTION_COUNTS.items()}


def option_text(question, labels, i, label):
    crit = question.get("criteria")
    if isinstance(crit, dict):
        if label in crit:
            return crit[label]
        if label in ("yes", "no"):
            return crit.get("true" if label == "yes" else "false", "")
    if isinstance(crit, list) and i < len(crit):
        c = crit[i]
        return c if isinstance(c, str) else json.dumps(c)
    return ""


def render(state, question, labels):
    ins = question.get("instructions", "") or ""
    lines = []
    for i, lab in enumerate(labels):
        desc = option_text(question, labels, i, lab)
        lines.append(f"{LETTERS[i]}. {lab}" + (f" - {desc}" if desc else ""))
    return f"{state}\n\n{ins}\n\n" + "\n".join(lines) + "\n\nAnswer:"


def read_letters(model, tok, prompt, n):
    """One forward pass. Returns (probs list for the first n letters, latency s)."""
    ids = _encode(tok, prompt, add_bos=True)
    t0 = time.perf_counter()
    logits = model(mx.array([ids]))
    mx.eval(logits)
    probs = mx.softmax(logits[0, -1, :].astype(mx.float32))
    mx.eval(probs)
    lat = time.perf_counter() - t0
    out = []
    for i in range(n):
        tid = _encode(tok, f" {LETTERS[i]}")[0]
        out.append(float(probs[tid]))
    return out, lat


def run_tier(model, tok, path, tier, reverse_noul=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    recs = []
    for r in rows:
        labels = list(r["labels"])
        q = r["question"]
        exp = r["expected"]
        if reverse_noul and q.get("type") == "noul" and len(labels) == 2:
            labels = labels[::-1]
        prompt = render(r["state"], q, labels)
        try:
            ps, lat = read_letters(model, tok, prompt, len(labels))
        except Exception as e:  # noqa: BLE001
            recs.append({"id": r["id"], "error": str(e)})
            continue
        tot = sum(ps) or 1.0
        share = [p / tot for p in ps]
        best = max(range(len(ps)), key=lambda i: ps[i])
        picked = labels[best]
        correct = str(picked) == str(exp)
        ev = None
        if q.get("type") == "score":
            ev = sum(i * s for i, s in enumerate(share))
            correct = (round(ev) == exp)
        recs.append({
            "id": r["id"], "tier": tier, "family": r["family"],
            "type": q.get("type"), "n_options": len(labels),
            "labels": labels, "expected": exp, "picked": picked,
            "correct": correct, "confidence": share[best], "raw_probs": ps, "share": share,
            "ev": ev, "latency_s": lat, "state_chars": len(r["state"]),
        })
    return recs


def ece_top_label(pairs, n_bins=10):
    bins = [{"n": 0, "cs": 0.0, "cor": 0} for _ in range(n_bins)]
    for conf, ok in pairs:
        c = min(max(float(conf), 0.0), 1.0)
        b = bins[min(int(c * n_bins), n_bins - 1)]
        b["n"] += 1
        b["cs"] += c
        b["cor"] += 1 if ok else 0
    n = sum(b["n"] for b in bins)
    if not n:
        return None, []
    ece = 0.0
    detail = []
    for i, b in enumerate(bins):
        if b["n"]:
            acc = b["cor"] / b["n"]
            mc = b["cs"] / b["n"]
            ece += (b["n"] / n) * abs(acc - mc)
            detail.append({"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": b["n"],
                           "conf": mc, "acc": acc})
    return ece, detail


def cca(acc, chance):
    return max(0.0, min(100.0, 100 * (acc - chance) / (1 - chance)))


def main():
    model, tok = load(MODEL)
    tiers = [
        ("easy", DATASETS + "/easy.jsonl"),
        ("standard", DATASETS + "/original.jsonl"),
        ("hard", DATASETS + "/hard.jsonl"),
    ]
    all_recs = {}
    for tier, path in tiers:
        t0 = time.time()
        recs = [r for r in run_tier(model, tok, path, tier) if "error" not in r]
        all_recs[tier] = recs
        print(f"[{tier}] {len(recs)} items in {time.time()-t0:.1f}s", flush=True)

    # option-order robustness on noul items (JevBench issue #40 failure mode)
    rev = []
    for tier, path in tiers:
        rev += [r for r in run_tier(model, tok, path, tier, reverse_noul=True) if "error" not in r]
    noul_rev = [r for r in rev if r["type"] == "noul"]
    noul_fwd = [r for t, rs in all_recs.items() for r in rs if r["type"] == "noul"]
    all_recs["reversed_noul"] = noul_rev

    json.dump({"tiers": all_recs, "noul_forward": noul_fwd, "noul_reversed": noul_rev},
              open(OUT, "w"), indent=1)

    print("\n" + "=" * 76)
    print("MiniCPM5-2B-MLX (4-bit), JevBench public items, letter-probability readout")
    print("=" * 76)
    intel_num = intel_den = 0.0
    tier_acc = {}
    print(f"{'tier':<10} {'n':>4} {'accuracy':>10} {'chance':>8} {'chance-corrected':>18}")
    for tier, _ in tiers:
        recs = all_recs[tier]
        acc = sum(r["correct"] for r in recs) / len(recs)
        tier_acc[tier] = acc
        ch = TIER_CHANCES[tier]
        cc = cca(acc, ch)
        intel_num += TIER_WEIGHTS[tier] * cc
        intel_den += TIER_WEIGHTS[tier]
        print(f"{tier:<10} {len(recs):>4} {acc*100:>9.1f}% {ch*100:>7.1f}% {cc:>17.1f}")
    intelligence = intel_num / intel_den
    print(f"{'WEIGHTED':<10} {'':>4} {'':>10} {'':>8} {intelligence:>17.1f}   <- intelligence axis")

    hard = all_recs["hard"]
    ece, detail = ece_top_label([(r["confidence"], r["correct"]) for r in hard])
    print(f"\nhard-tier ECE (top-label, 10 bins) = {ece:.4f}")
    for d in detail:
        print(f"   [{d['lo']:.1f},{d['hi']:.1f}) n={d['n']:>3} conf={d['conf']:.3f} acc={d['acc']:.3f}")

    lat = sorted(r["latency_s"] for t, _ in tiers for r in all_recs[t])
    p50 = lat[int(0.5 * (len(lat) - 1))]
    p95 = lat[int(0.95 * (len(lat) - 1))]
    adj50, adj95 = p50 * 2 + 0.15, p95 * 2 + 0.15
    sp = (max(0, min(100, 100 - 20 * math.log10(adj50 / 0.1))) +
          max(0, min(100, 100 - 20 * math.log10(adj95 / 0.1)))) / 2
    print(f"\nlatency raw p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms  "
          f"-> adjusted x2+0.15s (their own-server rule): p50={adj50:.2f}s p95={adj95:.2f}s")
    print(f"speed axis = {sp:.1f}")

    cal = max(0.0, 100 * (1 - ece / 0.5))
    print(f"calibration axis (ECE only) = {cal:.1f}")

    print("\noption-order robustness (noul, 2 options):")
    if noul_fwd and noul_rev:
        a = sum(r["correct"] for r in noul_fwd) / len(noul_fwd)
        b = sum(r["correct"] for r in noul_rev) / len(noul_rev)
        print(f"  forward order : {a*100:.1f}%  (n={len(noul_fwd)})")
        print(f"  reversed order: {b*100:.1f}%  (n={len(noul_rev)})")
        print(f"  delta = {(a-b)*100:+.1f} points")

    print("\naccuracy by family (hard tier):")
    fams = {}
    for r in hard:
        fams.setdefault(r["family"], []).append(r["correct"])
    for f, v in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        print(f"  {f:<18} {sum(v)}/{len(v)} = {sum(v)/len(v)*100:5.1f}%")

    summary = {"intelligence": intelligence, "tier_acc": tier_acc,
               "chance": TIER_CHANCES, "hard_ece": ece, "speed": sp, "calibration": cal,
               "p50_s": p50, "p95_s": p95, "adj_p50_s": adj50, "adj_p95_s": adj95}
    json.dump(summary, open("benchmarks/results/jevbench_minicpm_summary.json", "w"), indent=1)
    print(f"\nraw results -> {OUT}")


if __name__ == "__main__":
    main()
