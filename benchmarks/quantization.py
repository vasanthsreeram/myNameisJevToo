#!/usr/bin/env python3
"""
mcpm_quant_compare.py - does 4-bit quantization break the probabilities?

Runs the same Test-A candidate set through the bf16 checkpoint and the official
MLX 4-bit checkpoint and compares per-candidate mean log-probabilities.

If the two disagree materially, the "surgical" score is a property of the
quantization, not of the model - which matters if this becomes a deployed scorer.
"""
import json
import math
import sys

import mlx.core as mx
from mlx_lm import load

from _shared_ import score_continuation  # noqa: E402
from token_scoring import ITEMS  # noqa: E402


def run(path):
    model, tok = load(path)
    out = []
    for prompt, correct, wrongs in ITEMS:
        for c in [correct] + wrongs:
            s = score_continuation(model, tok, prompt, c)
            out.append({"prompt": prompt, "answer": c, "mean_logprob": s["mean_logprob"],
                        "is_correct": c == correct})
    return out


def main():
    bf16 = run(sys.argv[1] if len(sys.argv) > 1 else "/Users/admin/models/MiniCPM5-2B")
    q4 = run(sys.argv[2] if len(sys.argv) > 2 else "/Users/admin/models/MiniCPM5-2B-MLX")

    assert len(bf16) == len(q4)
    diffs = [abs(a["mean_logprob"] - b["mean_logprob"]) for a, b in zip(bf16, q4)]
    # rank correlation over all 100 candidates
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rb, rq = ranks([x["mean_logprob"] for x in bf16]), ranks([x["mean_logprob"] for x in q4])
    mb, mq = sum(rb) / len(rb), sum(rq) / len(rq)
    num = sum((a - mb) * (b - mq) for a, b in zip(rb, rq))
    den = math.sqrt(sum((a - mb) ** 2 for a in rb) * sum((b - mq) ** 2 for b in rq))
    rho = num / den

    def top1(rows):
        hits = 0
        for i in range(0, len(rows), 4):
            grp = rows[i:i + 4]
            best = max(grp, key=lambda r: r["mean_logprob"])
            hits += best["is_correct"]
        return hits

    print(f"candidates compared: {len(bf16)}")
    print(f"top-1 accuracy  bf16 = {top1(bf16)}/25   MLX-4bit = {top1(q4)}/25")
    print(f"Spearman rank correlation of mean logprob = {rho:.4f}")
    print(f"|delta mean logprob| : mean={sum(diffs)/len(diffs):.4f}  max={max(diffs):.4f} nats")
    flips = sum(1 for i in range(0, len(bf16), 4)
                if max(bf16[i:i+4], key=lambda r: r["mean_logprob"])["answer"]
                != max(q4[i:i+4], key=lambda r: r["mean_logprob"])["answer"])
    print(f"items where the argmax answer CHANGED under 4-bit: {flips}/25")
    with open("benchmarks/results/mcpm_quant_compare.json", "w") as f:
        json.dump({"spearman": rho, "mean_abs_delta": sum(diffs)/len(diffs), "max_abs_delta": max(diffs),
                   "top1_bf16": top1(bf16), "top1_q4": top1(q4), "argmax_flips": flips,
                   "bf16": bf16, "q4": q4}, f, indent=1)


if __name__ == "__main__":
    main()
