#!/usr/bin/env python3
"""mcpm_cached_check.py - prove the cached path equals the naive path, then time both."""
import statistics
import sys
import time

from latency import STATE, CANDS  # noqa: E402
from _shared_ import score_candidates_cached, score_continuation  # noqa: E402
from mlx_lm import load  # noqa: E402


def check(path, label):
    model, tok = load(path)
    ntok = len(tok.encode(STATE, add_special_tokens=False))

    naive = [score_continuation(model, tok, STATE, c) for c in CANDS]
    cached = score_candidates_cached(model, tok, STATE, CANDS)

    print(f"--- {label} (state = {ntok} tokens) ---")
    diffs = [abs(a["mean_logprob"] - b["mean_logprob"]) for a, b in zip(naive, cached)]
    print(f"equivalence  max |delta mean logprob| = {max(diffs):.8f} nats  "
          f"({'IDENTICAL' if max(diffs) < 1e-4 else 'MISMATCH'})")
    print(f"equivalence  argmax same: "
          f"{max(range(4), key=lambda i: naive[i]['mean_logprob']) == max(range(4), key=lambda i: cached[i]['mean_logprob'])}")

    for tag, fn in (("naive  ", lambda: [score_continuation(model, tok, STATE, c) for c in CANDS]),
                    ("cached ", lambda: score_candidates_cached(model, tok, STATE, CANDS))):
        fn()  # warmup
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"{tag} 4-way decision: median {statistics.median(ts):7.1f} ms  min {min(ts):7.1f} ms")
    print()
    return cached


if __name__ == "__main__":
    b = check("/Users/admin/models/MiniCPM5-2B", "bf16")
    q = check("/Users/admin/models/MiniCPM5-2B-MLX", "MLX 4-bit")
    print("cached scores, bf16  :", [(x["continuation"], round(x["mean_logprob"], 4)) for x in b])
    print("cached scores, 4-bit :", [(x["continuation"], round(x["mean_logprob"], 4)) for x in q])
