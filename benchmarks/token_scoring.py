#!/usr/bin/env python3
"""
mcpm_verify.py - Is MiniCPM5-2B usable as a "given answer -> how likely is it" scorer?

Three measurements, all on real forward passes (no sampling):

  TEST A  Answer ranking       - for each prompt, rank 4 candidate answers by
                                 length-normalised logprob. Is the correct one #1?
  TEST B  Token-level anomaly  - corrupt one token in a fluent passage. Does the
                                 score localise the corruption?
  TEST C  Separation (AUC)     - treat every scored token as a binary event
                                 (correct-answer token vs wrong-answer token) and
                                 compute AUC of p. 0.5 = useless, 1.0 = perfect.

Writes a JSON artifact with every raw number.
"""
import json
import math
import sys

import mlx.core as mx
from mlx_lm import load

from _shared_ import score_continuation, score_tokens, _encode  # noqa: E402

# ---------------------------------------------------------------- TEST A data
ITEMS = [
    ("The capital of Australia is", " Canberra", [" Sydney", " Melbourne", " Perth"]),
    ("The chemical symbol for gold is", " Au", [" Ag", " Gd", " Go"]),
    ("The largest planet in the solar system is", " Jupiter", [" Saturn", " Neptune", " Mars"]),
    ("At sea level, water boils at", " 100 degrees Celsius", [" 90 degrees Celsius", " 212 degrees Celsius", " 80 degrees Celsius"]),
    ("The author of the novel Pride and Prejudice is", " Jane Austen", [" Charlotte Bronte", " Emily Bronte", " Virginia Woolf"]),
    ("Singapore gained independence in the year", " 1965", [" 1963", " 1957", " 1971"]),
    ("The longest river in the world is generally considered the", " Nile", [" Amazon", " Yangtze", " Mississippi"]),
    ("The currency of Japan is the", " yen", [" yuan", " won", " baht"]),
    ("The human heart has", " four chambers", [" two chambers", " three chambers", " six chambers"]),
    ("The programming language created by Guido van Rossum is", " Python", [" Ruby", " Perl", " Java"]),
    ("The planet closest to the Sun is", " Mercury", [" Venus", " Mars", " Earth"]),
    ("The chemical formula for table salt is", " NaCl", [" KCl", " HCl", " NaHCO3"]),
    ("The tallest mountain above sea level is", " Mount Everest", [" K2", " Kangchenjunga", " Denali"]),
    ("The speed of light in a vacuum is approximately 300,000", " km per second", [" metres per second", " miles per hour", " cm per second"]),
    ("The first person to walk on the Moon was", " Neil Armstrong", [" Buzz Aldrin", " Yuri Gagarin", " Michael Collins"]),
    ("The largest organ in the human body is the", " skin", [" liver", " brain", " lung"]),
    ("The most abundant gas in Earth's atmosphere is", " nitrogen", [" oxygen", " carbon dioxide", " argon"]),
    ("The co-founder of Microsoft is", " Bill Gates", [" Steve Jobs", " Jeff Bezos", " Larry Page"]),
    ("The capital of Canada is", " Ottawa", [" Toronto", " Vancouver", " Montreal"]),
    ("Binary 1010 in decimal is", " 10", [" 8", " 12", " 14"]),
    ("The number of sides of a hexagon is", " six", [" five", " seven", " eight"]),
    ("The boiling point of nitrogen in Celsius is about", " -196", [" -100", " 0", " 100"]),
    ("The Great Barrier Reef lies off the coast of", " Australia", [" Brazil", " India", " Mexico"]),
    ("The metric prefix for one thousandth is", " milli", [" micro", " kilo", " centi"]),
    ("The square root of 144 is", " 12", [" 14", " 16", " 11"]),
]

CLEAN = "The quick brown fox jumped over the lazy dog and then ran through the tall grass beside the quiet river."
CORRUPT = [
    ("fox", "telescope"),
    ("dog", "helicopter"),
    ("river", "sandwich"),
    ("grass", "algebra"),
]


def auc(pos, neg):
    """Rank-based AUC (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    pairs = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    pairs.sort(key=lambda x: x[0])
    ranks, i = {}, 0
    vals = [p[0] for p in pairs]
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_pos = sum(ranks[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(xs, ys, iters=6000, lr=0.5):
    """1-D Platt scaling: p = sigmoid(a*x + b). Plain gradient descent."""
    a, b = 1.0, 0.0
    n = max(len(xs), 1)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            e = _sigmoid(a * x + b) - y
            ga += e * x
            gb += e
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/admin/models/MiniCPM5-2B"
    model, tok = load(path)
    report = {"model": path, "test_a": [], "test_b": [], "test_c": {}}

    print("=" * 78)
    print("TEST A - answer ranking (correct vs distractors, length-normalised)")
    print("=" * 78)
    hits = 0
    pos_tokens, neg_tokens = [], []
    ans_pts = []
    for item_idx, (prompt, correct, wrongs) in enumerate(ITEMS):
        cands = [correct] + wrongs
        scored = [score_continuation(model, tok, prompt, c) for c in cands]
        order = sorted(range(len(cands)), key=lambda i: -scored[i]["mean_logprob"])
        winner = order[0]
        ok = winner == 0
        hits += ok
        margin = scored[0]["mean_logprob"] - max(
            s["mean_logprob"] for i, s in enumerate(scored) if i != 0
        )
        report["test_a"].append({
            "prompt": prompt, "correct": correct,
            "rank_of_correct": order.index(0) + 1,
            "correct_mean_logprob": scored[0]["mean_logprob"],
            "best_wrong": cands[order[1]], "best_wrong_mean_logprob": scored[order[1]]["mean_logprob"],
            "margin": margin,
            "per_candidate": [{"answer": c, "mean_logprob": s["mean_logprob"], "ppl": s["perplexity"]}
                              for c, s in zip(cands, scored)],
        })
        pos_tokens += [t["p"] for t in scored[0]["tokens"]]
        for i in range(1, len(cands)):
            neg_tokens += [t["p"] for t in scored[i]["tokens"]]
        ans_pts.append((scored[0]["geometric_prob"], 1, item_idx))
        for i in range(1, len(cands)):
            ans_pts.append((scored[i]["geometric_prob"], 0, item_idx))
        flag = "OK " if ok else "MISS"
        print(f"{flag} {prompt[:46]:<46} correct ppl={scored[0]['perplexity']:>8.2f} "
              f"margin={margin:>+6.3f}  bestwrong={cands[order[1]]!r}")
    acc = hits / len(ITEMS)
    print(f"\nTEST A top-1 accuracy: {hits}/{len(ITEMS)} = {acc*100:.1f}%")
    report["test_a_accuracy"] = acc

    print()
    print("=" * 78)
    print("TEST B - token-level anomaly localisation (single corrupt token)")
    print("=" * 78)
    clean_ids = _encode(tok, CLEAN, add_bos=True)
    clean_rows = score_tokens(model, tok, clean_ids)
    clean_min = min(r["p"] for r in clean_rows)
    clean_med = sorted(r["p"] for r in clean_rows)[len(clean_rows) // 2]
    print(f"clean passage: worst-token p = {clean_min:.6g}, median p = {clean_med:.6g} "
          f"({len(clean_rows)} scored positions)")
    corrupt_ps = []
    for good, bad in CORRUPT:
        corrupt = CLEAN.replace(good, bad, 1)
        cids = _encode(tok, corrupt, add_bos=True)
        rows = score_tokens(model, tok, cids)
        n = min(len(clean_ids), len(cids))
        pos = next((i for i in range(1, n) if clean_ids[i] != cids[i]), None)
        if pos is None:
            print(f"  skipped {good}->{bad}: no token divergence found")
            continue
        ridx = pos - 1  # rows[i] is the score for ids[i+1]
        p_c = rows[ridx]["p"]
        p_ok = clean_rows[ridx]["p"] if ridx < len(clean_rows) else float("nan")
        corrupt_ps.append(p_c)
        worst_anywhere = min(r["p"] for r in rows)
        rec = {
            "clean_word": good, "corrupt_word": bad, "position": pos,
            "p_clean": p_ok, "p_corrupt": p_c,
            "pct_outside_corrupt": rows[ridx]["pct_outside"],
            "rank_corrupt": rows[ridx]["rank"],
            "surprisal_clean": clean_rows[ridx]["surprisal"] if ridx < len(clean_rows) else None,
            "surprisal_corrupt": rows[ridx]["surprisal"],
            "is_worst_in_sentence": worst_anywhere == p_c,
        }
        report["test_b"].append(rec)
        print(f"{good:>8} -> {bad:<11} pos={pos:<3} p_clean={p_ok:.5g} -> p_corrupt={p_c:.3e}  "
              f"rank={rec['rank_corrupt']:>6d}/{130560}  outside={rec['pct_outside_corrupt']:.5f}%  "
              f"delta_surprisal={rec['surprisal_corrupt'] - rec['surprisal_clean']:+.2f} nats")
    if corrupt_ps:
        band = (clean_min, max(corrupt_ps))
        report["test_b_summary"] = {"clean_min_p": clean_min, "corrupt_max_p": max(corrupt_ps),
                                    "clean_median_p": clean_med}
        print(f"\nseparation: every corrupted token p <= {max(corrupt_ps):.3e}, "
              f"cleanest passage token p >= {clean_min:.3e}")
        print(f"a single fixed cut-off (e.g. p < 1e-4) separates them: "
              f"{'YES' if max(corrupt_ps) < clean_min else 'NO - overlap'}")

    print()
    print("=" * 78)
    print("TEST C - separation: does p discriminate correct-answer tokens from wrong?")
    print("=" * 78)
    a = auc(pos_tokens, neg_tokens)
    print(f"tokens: {len(pos_tokens)} correct-side, {len(neg_tokens)} wrong-side")
    print(f"median p (correct-side tokens) = {sorted(pos_tokens)[len(pos_tokens)//2]:.4f}")
    print(f"median p (wrong-side tokens)   = {sorted(neg_tokens)[len(neg_tokens)//2]:.4f}")
    print(f"AUC = {a:.4f}   (0.5 = coin flip, 1.0 = perfect separation)")
    report["test_c"] = {
        "auc": a, "n_pos": len(pos_tokens), "n_neg": len(neg_tokens),
        "median_p_pos": sorted(pos_tokens)[len(pos_tokens)//2],
        "median_p_neg": sorted(neg_tokens)[len(neg_tokens)//2],
    }

    # ---------------------------------------------------------- TEST D
    print()
    print("=" * 78)
    print("TEST D - can the raw likelihood become a CALIBRATED probability (noul)?")
    print("=" * 78)
    xs = [math.log10(max(p, 1e-12)) for p, _, _ in ans_pts]
    ys = [y for _, y, _ in ans_pts]
    groups = [g for _, _, g in ans_pts]
    n_items = len(ITEMS)
    folds = [[i for i in range(n_items) if i % 5 == k] for k in range(5)]
    oof = [None] * len(ys)
    for k in range(5):
        te = set(folds[k])
        tr = [i for i, g in enumerate(groups) if g not in te]
        a, b = fit_platt([xs[i] for i in tr], [ys[i] for i in tr])
        for i, g in enumerate(groups):
            if g in te:
                oof[i] = _sigmoid(a * xs[i] + b)

    def brier(ps, ls):
        return sum((p - l) ** 2 for p, l in zip(ps, ls)) / len(ls)

    def ece(ps, ls, bins=10):
        tot, out = len(ls), 0.0
        for k in range(bins):
            lo, hi = k / bins, (k + 1) / bins
            idx = [i for i, p in enumerate(ps) if (lo < p <= hi) or (k == 0 and p <= lo)]
            if not idx:
                continue
            conf = sum(ps[i] for i in idx) / len(idx)
            acc = sum(ls[i] for i in idx) / len(idx)
            out += (len(idx) / tot) * abs(conf - acc)
        return out

    a_ans = auc([p for p, l, _ in ans_pts if l == 1], [p for p, l, _ in ans_pts if l == 0])
    raw_brier = brier([min(max(p, 0.0), 1.0) for p, _, _ in ans_pts], ys)
    cal_brier = brier(oof, ys)
    cal_ece = ece(oof, ys)
    acc_thresh = sum(1 for p, l in zip(oof, ys) if (p >= 0.5) == (l == 1)) / len(ys)
    # best achievable accuracy by sweeping a cut-off on the raw log10-likelihood
    grid = sorted(set(xs))
    mids = [(grid[i] + grid[i + 1]) / 2 for i in range(len(grid) - 1)] + [min(grid) - 1, max(grid) + 1]
    best = max(sum(1 for x, l in zip(xs, ys) if (x >= t) == (l == 1)) / len(ys) for t in mids)
    print(f"answer-level points: {len(ys)} ({sum(ys)} positive / {len(ys)-sum(ys)} negative), 5-fold CV grouped by item")
    print(f"answer-level AUC (raw geometric p)     = {a_ans:.4f}")
    print(f"best achievable accuracy (cut-off on log-likelihood, in-sample) = {best*100:.1f}%")
    print(f"Brier, raw p                           = {raw_brier:.4f}")
    print(f"Brier, Platt-calibrated (out-of-fold)  = {cal_brier:.4f}")
    print(f"ECE   Platt-calibrated (10 bins)       = {cal_ece:.4f}")
    print(f"accuracy @ p>=0.5 (out-of-fold)        = {acc_thresh*100:.1f}%")
    report["test_d"] = {
        "n_points": len(ys), "n_pos": int(sum(ys)),
        "answer_auc": a_ans, "best_threshold_acc": best,
        "brier_raw": raw_brier, "brier_platt_oof": cal_brier,
        "ece_platt_oof": cal_ece, "acc_at_half_oof": acc_thresh,
        "points": [{"log10p": x, "label": y, "oof_calibrated": o} for x, y, o in zip(xs, ys, oof)],
    }

    out = "benchmarks/results/mcpm_verify_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nraw numbers written to {out}")


if __name__ == "__main__":
    main()
