#!/usr/bin/env python3
"""mcpm_length_sweep.py - decision latency vs state length (1 prefill + 4 candidates)."""
import statistics
import sys
import time

from latency import CANDS, STATE  # noqa: E402
from _shared_ import score_candidates_cached_ids, _encode  # noqa: E402
from mlx_lm import load  # noqa: E402


def measure(model, tok, ids, runs=8):
    score_candidates_cached_ids(model, tok, ids, CANDS)  # warmup
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        score_candidates_cached_ids(model, tok, ids, CANDS)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts)


def main():
    base = STATE
    for path, label in (("/Users/admin/models/MiniCPM5-2B", "bf16"),
                        ("/Users/admin/models/MiniCPM5-2B-MLX", "4bit")):
        model, tok = load(path)
        bos = [tok.bos_token_id]
        full = _encode(tok, base * 8)
        print(f"=== {label} ===")
        print(f"{'state tokens':>13} {'1 prefill+4 cands':>18} {'best':>9} {'vs Jev 70-500ms':>17}")
        for L in (50, 150, 300, 600, 1200, 2400):
            ids = bos + full[:L]
            med, mn = measure(model, tok, ids)
            verdict = "inside" if med <= 500 else ("near" if med <= 900 else "slower")
            print(f"{len(ids):>13} {med:>15.1f} ms {mn:>6.1f} ms {verdict:>17}")
        print()


if __name__ == "__main__":
    main()
