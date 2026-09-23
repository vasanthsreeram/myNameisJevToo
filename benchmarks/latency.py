#!/usr/bin/env python3
"""mcpm_latency.py - how fast is a Jev-style decision on MiniCPM5-2B?

Realistic shape: a ~300-token application state + 4 candidate answers,
scored in one pass each. Reports median over N timed runs after warmup.
"""
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load

from _shared_ import score_continuation  # noqa: E402

STATE = (
    "Support ticket 44821. Customer: Priya, on the Pro annual plan since March. "
    "Message: I was charged twice this month, once on the 3rd and again on the 5th, "
    "both for 49 dollars. I have already emailed twice and nobody has replied. "
    "I need this refunded today or I am cancelling and disputing the charge with my bank. "
    "Account history: plan upgraded in June, two successful payments in July and August, "
    "one failed payment attempt on the 4th that was retried and succeeded on the 5th. "
    "No prior refunds on this account. Support sentiment trend: two neutral tickets in May, "
    "one positive in June, one negative in July about invoice clarity. "
) * 2

CANDS = [" billing", " refund", " account", " feature"]


def bench(path, label, runs=12):
    model, tok = load(path)
    # warmup
    score_continuation(model, tok, STATE, CANDS[0])
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for c in CANDS:
            score_continuation(model, tok, STATE, c)
        ts.append((time.perf_counter() - t0) * 1000)
    ntok = len(tok.encode(STATE, add_special_tokens=False))
    print(f"{label:<12} state={ntok} tokens  "
          f"4-way decision median={statistics.median(ts):7.1f} ms  "
          f"min={min(ts):7.1f}  max={max(ts):7.1f}   "
          f"per-pass={statistics.median(ts)/4:6.1f} ms")
    return statistics.median(ts)


if __name__ == "__main__":
    a = bench("/Users/admin/models/MiniCPM5-2B", "bf16")
    b = bench("/Users/admin/models/MiniCPM5-2B-MLX", "MLX-4bit")
    print(f"\n4-bit speedup: {a/b:.2f}x")
