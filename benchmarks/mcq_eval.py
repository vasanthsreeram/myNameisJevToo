#!/usr/bin/env python3
"""
mcpm_mcq_eval.py - does reading the LETTER probability (A/B/C/D) actually work?

Model: openbmb/MiniCPM5-2B-MLX, 4-bit affine (Apple Silicon / MLX).

Methods, same items, same option order:
  L1  letter readout, plain scaffold          "\n\nAnswer:"        (1 forward pass)
  L2  letter readout + explicit instruction                        (1 forward pass)
  T   text readout: p of each option's own text after "Answer:"    (1 prefill + 4 cached passes)
  L1d debiased: letter p divided by a leave-one-out letter prior
  H   hybrid: z-scored letter logprob + z-scored text logprob

Also reports position bias (accuracy split by where the correct option sits)
and the raw letter prior, since a strong prior would explain inflated accuracy.
"""
import random
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load

from letter_readout import choice_probs  # noqa: E402
from _shared_ import score_candidates_cached_ids, _encode  # noqa: E402
from token_scoring import ITEMS as HAND_ITEMS  # noqa: E402

LABELS = (" A", " B", " C", " D")

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo", "Kenya": "Nairobi",
            "Norway": "Oslo", "Thailand": "Bangkok", "Peru": "Lima", "Portugal": "Lisbon",
            "Vietnam": "Hanoi", "Spain": "Madrid", "Greece": "Athens", "Turkey": "Ankara"}
ELEMENTS = {"gold": "Au", "silver": "Ag", "iron": "Fe", "copper": "Cu",
            "sodium": "Na", "potassium": "K", "lead": "Pb", "tin": "Sn"}


def generated_items(seed=7):
    rng = random.Random(seed)
    items = []
    caps = list(CAPITALS.items())
    for country, cap in caps:
        pool = [c for c2, c in caps if c != cap]
        wrongs = rng.sample(pool, 3)
        items.append((f"What is the capital city of {country}?", f" {cap}",
                      [f" {w}" for w in wrongs]))
    els = list(ELEMENTS.items())
    for name, sym in els:
        pool = [s for n2, s in els if s != sym]
        wrongs = rng.sample(pool, 3)
        items.append((f"What is the chemical symbol for {name}?", f" {sym}",
                      [f" {w}" for w in wrongs]))
    for n in range(11, 21):
        sq = n * n
        pool = set()
        while len(pool) < 3:
            cand = sq + rng.choice([-20, -11, -9, -4, 7, 9, 11, 18, 21, 30])
            if cand != sq and cand > 0:
                pool.add(cand)
        items.append((f"What is {n} squared?", f" {sq}", [f" {c}" for c in sorted(pool)]))
    return items


def build_cases(seed=1234):
    cases = []
    for i, (q, correct, wrongs) in enumerate(list(HAND_ITEMS) + generated_items()):
        opts = [correct.strip()] + [w.strip() for w in wrongs]
        random.Random(seed + i).shuffle(opts)
        cases.append({"q": q, "opts": opts, "correct_idx": opts.index(correct.strip())})
    return cases


def state_plain(q, opts):
    body = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
    return f"{q}\n\n{body}\n\nAnswer:"


def state_instruct(q, opts):
    body = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
    return (f"{q}\n\n{body}\n\n"
            f"Answer with a single letter (A, B, C or D).\nAnswer:")


def zscore(v):
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
    return [(x - m) / sd for x in v]


def main():
    model, tok = load("/Users/admin/models/MiniCPM5-2B-MLX")
    cases = build_cases()
    n = len(cases)

    recs = []
    t_letter, t_text = [], []
    for c in cases:
        q, opts = c["q"], c["opts"]
        s1 = state_plain(q, opts)
        t0 = time.perf_counter()
        per1, _ = choice_probs(model, tok, s1, LABELS)
        t_letter.append((time.perf_counter() - t0) * 1000)
        s2 = state_instruct(q, opts)
        per2, _ = choice_probs(model, tok, s2, LABELS)
        t0 = time.perf_counter()
        cands = [" " + o for o in opts]
        scored = score_candidates_cached_ids(model, tok, _encode(tok, s1, add_bos=True), cands)
        t_text.append((time.perf_counter() - t0) * 1000)
        recs.append({
            "ci": c["correct_idx"],
            "p1": [per1[l]["p"] for l in LABELS],
            "p2": [per2[l]["p"] for l in LABELS],
            "ptxt": [s["mean_logprob"] for s in scored],
        })

    def evaluate(name, scorefn):
        hit, by_pos = 0, {i: [0, 0] for i in range(4)}
        p_correct, p_picked = [], []
        for r in recs:
            sc = scorefn(r)
            pick = max(range(4), key=lambda i: sc[i])
            ok = pick == r["ci"]
            hit += ok
            by_pos[r["ci"]][0] += ok
            by_pos[r["ci"]][1] += 1
            p_correct.append(sc[r["ci"]])
            p_picked.append(sc[pick])
        return {"name": name, "acc": hit / n, "hit": hit, "by_pos": by_pos,
                "p_correct": statistics.median(p_correct),
                "p_picked": statistics.median(p_picked)}

    # leave-one-out letter prior, to debias without leaking the item itself
    def loo_prior(r, which, k=0.25):
        """Conjugate-ish prior from every OTHER item, with a small uniform floor."""
        vals = [0.0] * 4
        for o in recs:
            if o is r:
                continue
            for i in range(4):
                vals[i] += o[which][i]
        tot = sum(vals) or 1.0
        return [(v / tot) + k for v in vals]

    def debiased(r):
        pr = loo_prior(r, "p1")
        return [r["p1"][i] / pr[i] for i in range(4)]

    def debiased2(r):
        pr = loo_prior(r, "p2")
        return [r["p2"][i] / pr[i] for i in range(4)]

    def hybrid(r):
        a, b = zscore(r["p1"]), zscore(r["ptxt"])
        return [a[i] + b[i] for i in range(4)]

    rows = [
        evaluate("L1 letter readout, plain scaffold", lambda r: r["p1"]),
        evaluate("L2 letter readout + instruction", lambda r: r["p2"]),
        evaluate("L1d letter readout, prior-debiased", debiased),
        evaluate("L2d letter+instr, prior-debiased", debiased2),
        evaluate("T  text readout after 'Answer:'", lambda r: r["ptxt"]),
        evaluate("H  hybrid letter + text", hybrid),
    ]

    print(f"items: {n} (25 hand-written + {n-25} generated)   model: openbmb/MiniCPM5-2B-MLX 4-bit")
    print(f"correct-option position shuffled deterministically\n")
    print(f"{'method':<40} {'accuracy':>13} {'median score':>14}")
    for r in rows:
        print(f"{r['name']:<40} {r['hit']:>3}/{n} {r['acc']*100:6.1f}% {r['p_correct']:>14.6f}")

    print(f"\naccuracy by correct-option position (position bias):")
    print(f"{'method':<40} {'A':>7} {'B':>7} {'C':>7} {'D':>7}")
    for r in rows:
        cells = [f"{r['by_pos'][i][0]}/{r['by_pos'][i][1]}" if r['by_pos'][i][1] else "-" for i in range(4)]
        print(f"{r['name']:<40} {cells[0]:>7} {cells[1]:>7} {cells[2]:>7} {cells[3]:>7}")

    print("\nletter prior, L1 (mean p per label across all items):")
    means = [statistics.mean([r["p1"][i] for r in recs]) for i in range(4)]
    tot = sum(means)
    for i, lab in enumerate(LABELS):
        print(f"  {lab!r}: p={means[i]:.6f}  share={means[i]/tot*100:5.2f}%   "
              f"{'#' * int(means[i]/tot*50)}")

    # ---- calibration: when it says 90%, is it right 90% of the time?
    print("\ncalibration - confidence = p of the picked label, vs empirical accuracy:")
    print(f"{'method':<8} {'bin':>14} {'n':>4} {'mean conf':>10} {'accuracy':>9} {'gap':>8}")
    for which, tag in (("p1", "L1"), ("p2", "L2")):
        pts = []
        for r in recs:
            pick = max(range(4), key=lambda i: r[which][i])
            pts.append((r[which][pick], pick == r["ci"]))
        ece, edges = 0.0, [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        for lo, hi in edges:
            grp = [q for q in pts if lo <= q[0] < hi]
            if not grp:
                continue
            conf = statistics.mean(p for p, _ in grp)
            acc = sum(1 for _, ok in grp if ok) / len(grp)
            ece += (len(grp) / len(pts)) * abs(conf - acc)
            print(f"{tag:<8} {f'[{lo:.2f},{hi:.2f})':>14} {len(grp):>4} {conf:>10.4f} "
                  f"{acc*100:>8.1f}% {conf - acc:>+8.4f}")
        conf_wrong = sum(1 for p, ok in pts if p >= 0.9 and not ok)
        print(f"{tag:<8} ECE = {ece:.4f}   confidently wrong (p>=0.90): {conf_wrong}/{len(pts)}")

    # ---- act / abstain curve: Jev's "gate" primitive
    print("\nabstain curve - act only when picked p clears a threshold:")
    print(f"{'threshold':>10} {'coverage':>10} {'accuracy on acted':>18} {'wrong acted':>12}")
    for which, tag in (("p2", "L2"),):
        for th in (0.0, 0.5, 0.7, 0.85, 0.95):
            acted = [(r[which][max(range(4), key=lambda i: r[which][i])],
                      max(range(4), key=lambda i: r[which][i]) == r["ci"]) for r in recs]
            sub = [a for a in acted if a[0] >= th]
            if not sub:
                continue
            acc = sum(1 for _, ok in sub if ok) / len(sub)
            wrong = sum(1 for _, ok in sub if not ok)
            print(f"{th:>10.2f} {len(sub)/len(acted)*100:>9.1f}% {acc*100:>17.1f}% {wrong:>12}")

    print(f"\nlatency per decision (4 choices, state ~60-90 tokens):")
    print(f"  letter readout   median {statistics.median(t_letter):7.1f} ms   (1 prefill, zero extra passes)")
    print(f"  text   readout   median {statistics.median(t_text):7.1f} ms   (1 prefill + 4 cached passes)")


if __name__ == "__main__":
    main()
