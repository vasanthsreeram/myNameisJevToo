#!/usr/bin/env python3
"""
jevbench_stateblind.py - the control that matters.

The jev-calibration-audit found that on MMLU-ProX, shown ONLY the options with the
state replaced by a placeholder, Jev still scored 0.383 (Korean) / 0.463 (English)
against a chance rate near 0.15 - "a third to a half of apparent multiple-choice
accuracy on this benchmark is recoverable from the option list alone."

So: run our model with the state redacted and see how much accuracy survives.
Anything that survives is option-list exploitation, not judgment.
"""
import json
import sys

from mlx_lm import load

from jevbench import MODEL, TIER_CHANCES, cca, read_letters, render  # noqa: E402

BLIND = "[state withheld]"
DATASETS = __import__("os").environ.get("JEVBENCH_DATA", "datasets/jevbench")


def main():
    model, tok = load(MODEL)
    tiers = [("easy", DATASETS + "/easy.jsonl"),
             ("standard", DATASETS + "/original.jsonl"),
             ("hard", DATASETS + "/hard.jsonl")]
    out = {}
    print(f"{'tier':<10} {'n':>4} {'state-blind acc':>16} {'chance':>8} {'recoverable':>13}")
    for tier, path in tiers:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        hit = 0
        recs = []
        for r in rows:
            labels, q = list(r["labels"]), r["question"]
            prompt = render(BLIND, q, labels)
            ps, _ = read_letters(model, tok, prompt, len(labels))
            best = max(range(len(ps)), key=lambda i: ps[i])
            ok = str(labels[best]) == str(r["expected"])
            if q.get("type") == "score":
                tot = sum(ps) or 1.0
                ev = sum(i * (p / tot) for i, p in enumerate(ps))
                ok = (round(ev) == r["expected"])
            hit += ok
            recs.append({"id": r["id"], "correct": ok, "picked": labels[best],
                         "expected": r["expected"]})
        acc = hit / len(rows)
        ch = TIER_CHANCES[tier]
        print(f"{tier:<10} {len(rows):>4} {acc*100:>15.1f}% {ch*100:>7.1f}% "
              f"{'':>13}")
        out[tier] = {"n": len(rows), "accuracy": acc, "chance": ch,
                     "chance_corrected": cca(acc, ch), "items": recs}

    # combine with the full-state numbers if available
    try:
        full = json.load(open("benchmarks/results/jevbench_minicpm_results.json"))["tiers"]
        print("\nleakage check - accuracy with state vs without:")
        print(f"{'tier':<10} {'with state':>12} {'without':>10} {'kept':>8} {'verdict':>26}")
        for tier, _ in tiers:
            f = full.get(tier) or []
            if not f:
                continue
            fa = sum(r["correct"] for r in f) / len(f)
            ba = out[tier]["accuracy"]
            kept = ba / fa * 100 if fa else 0
            verdict = "state carries the signal" if kept < 40 else \
                      ("PARTLY OPTION-LIST DRIVEN" if kept < 80 else "MOSTLY OPTION-LIST DRIVEN")
            print(f"{tier:<10} {fa*100:>11.1f}% {ba*100:>9.1f}% {kept:>7.0f}% {verdict:>26}")
    except FileNotFoundError:
        print("(full-state results not found yet)")

    json.dump(out, open("benchmarks/results/jevbench_minicpm_stateblind.json", "w"), indent=1)


if __name__ == "__main__":
    main()
