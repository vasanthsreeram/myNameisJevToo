#!/usr/bin/env python3
"""
mcpm_score.py - Surgical per-token scoring for MiniCPM5-2B (and any LlamaForCausalLM)
on Apple Silicon via MLX.

THE OPERATION (not autoregressive generation):
  Feed the FULL sequence [prompt + given_answer] in ONE forward pass.
  The causal mask means position t only sees tokens < t, so logits[:, t-1]
  IS the model's distribution for token t. We never sample. We just read.

  For every position where the given token is v:
      p        = softmax(logits)[v]              # probability of the given token
      rank     = #{u : logit_u > logit_v}        # 0-indexed rank, 0 = model's top pick
      pct_out  = 100 * (1 - p)                   # "% of mass that disagrees"
      mass_abv = sum{p_u : p_u > p_v}            # mass strictly above the token
      surprisal= -log(p)  (nats)
      margin   = logit_max - logit_v             # how far below the model's pick

  "How likely is the given answer" = mean logprob over its tokens (length-normalised),
  perplexity = exp(-mean logprob).

Usage:
  python3 mcpm_score.py --model PATH --text "..."            # token-by-token table
  python3 mcpm_score.py --model PATH --prompt P --cont C ... # score candidate answers
"""
import argparse
import json
import math
import sys

import mlx.core as mx
from mlx_lm import load


def _encode(tok, text, add_bos=False):
    """MLX TokenizerWrapper.encode() prepends BOS on EVERY call.
    Concatenating two encodes therefore injects a bogus <s> mid-sequence,
    which wrecks the score of the first continuation token. Always call with
    add_special_tokens=False and add BOS exactly once, at the very front.

    Some tokenizers define no BOS at all (Qwen3.5's bos_token_id is None);
    prepending it there produces [None] + ids, which throws later and looks
    like the model failed rather than the tokenizer.
    """
    ids = tok.encode(text, add_special_tokens=False)
    bos = getattr(tok, "bos_token_id", None)
    if add_bos and bos is not None:
        ids = [bos] + ids
    return ids


def load_model(path, quantized=None):
    model, tok = load(path)
    return model, tok


def _shift_logits(model, ids):
    """One forward pass. Returns logits predicting tokens 1..T-1 -> shape [T-1, V]."""
    logits = model(mx.array([ids]))
    mx.eval(logits)
    return logits[0, :-1, :]  # position i predicts ids[i+1]


def score_tokens(model, tok, ids):
    """Per-token diagnostic for ids[1:] (the continuation part is up to the caller)."""
    lp = _shift_logits(model, ids)
    targets = mx.array(ids[1:])
    probs = mx.softmax(lp.astype(mx.float32), axis=-1)
    mx.eval(probs)

    tgt_logit = mx.take_along_axis(lp, targets[:, None], axis=-1)[:, 0]
    tgt_prob = mx.take_along_axis(probs, targets[:, None], axis=-1)[:, 0]
    max_logit = mx.max(lp, axis=-1)
    rank = mx.sum((lp > tgt_logit[:, None]).astype(mx.int32), axis=-1)
    mass_above = mx.sum(mx.where(probs > tgt_prob[:, None], probs, mx.zeros_like(probs)), axis=-1)
    mx.eval(tgt_prob, rank, mass_above, max_logit)

    out = []
    for i, tid in enumerate(ids[1:]):
        p = float(tgt_prob[i])
        out.append({
            "token": tok.decode([tid]),
            "token_id": int(tid),
            "p": p,
            "logprob": math.log(max(p, 1e-12)),
            "rank": int(rank[i]),
            "pct_outside": 100.0 * (1.0 - p),
            "mass_above": float(mass_above[i]),
            "surprisal": -math.log(max(p, 1e-12)),
            "margin": float(max_logit[i] - tgt_logit[i]),
        })
    return out


def score_continuation(model, tok, prompt, continuation, length_norm=True):
    """Answer-given scoring: how likely is `continuation` as the completion of `prompt`?"""
    p_ids = _encode(tok, prompt, add_bos=True)
    c_ids = _encode(tok, continuation)
    ids = p_ids + c_ids
    per = score_tokens(model, tok, ids)
    cont = per[len(p_ids) - 1:]  # first continuation token is predicted at index len(p)-1
    if not cont:
        return None
    logps = [t["logprob"] for t in cont]
    total = sum(logps)
    n = len(cont)
    mean_lp = total / n if length_norm else total
    return {
        "prompt": prompt,
        "continuation": continuation,
        "n_tokens": n,
        "total_logprob": total,
        "mean_logprob": mean_lp,
        "perplexity": math.exp(-mean_lp),
        "geometric_prob": math.exp(mean_lp),
        "min_token_p": min(t["p"] for t in cont),
        "max_token_p": max(t["p"] for t in cont),
        "worst_token": min(cont, key=lambda t: t["p"]),
        "tokens": cont,
    }


def _row_scores(logits_row, tok_id, tok, cache_note=None):
    """Diagnostics for one distribution: logits_row is the model's prediction
    for a single position; tok_id is the token that was actually given."""
    probs = mx.softmax(logits_row.astype(mx.float32))
    tgt_logit = logits_row[tok_id]
    p = probs[tok_id]
    rank = mx.sum((logits_row > tgt_logit).astype(mx.int32))
    mass_above = mx.sum(mx.where(probs > p, probs, mx.zeros_like(probs)))
    mx.eval(p, rank, mass_above)
    pf = float(p)
    return {
        "token": tok.decode([tok_id]),
        "token_id": int(tok_id),
        "p": pf,
        "logprob": math.log(max(pf, 1e-12)),
        "rank": int(rank),
        "pct_outside": 100.0 * (1.0 - pf),
        "mass_above": float(mass_above),
        "surprisal": -math.log(max(pf, 1e-12)),
        "margin": float(mx.max(logits_row) - tgt_logit),
    }


def _pack(tok, prompt, continuation, cont):
    """Shared result shaping for both scoring paths."""
    logps = [t["logprob"] for t in cont]
    total, n = sum(logps), len(cont)
    mean_lp = total / n
    return {
        "prompt": prompt, "continuation": continuation, "n_tokens": n,
        "total_logprob": total, "mean_logprob": mean_lp,
        "perplexity": math.exp(-mean_lp), "geometric_prob": math.exp(mean_lp),
        "min_token_p": min(t["p"] for t in cont), "max_token_p": max(t["p"] for t in cont),
        "worst_token": min(cont, key=lambda t: t["p"]), "tokens": cont,
    }


def score_candidates_cached_ids(model, tok, p_ids, candidates):
    """Same as score_candidates_cached but takes pre-encoded prefix token ids."""
    import copy
    from mlx_lm.models.cache import make_prompt_cache

    pc = make_prompt_cache(model)
    pre = model(mx.array([p_ids]), cache=pc)
    mx.eval(pre)
    for c in pc:
        if getattr(c, "keys", None) is not None:
            mx.eval(c.keys, c.values)

    results = []
    for cont in candidates:
        c_ids = _encode(tok, cont)
        if not c_ids:
            continue
        pc_i = copy.deepcopy(pc)
        cl = model(mx.array([c_ids]), cache=pc_i)
        mx.eval(cl)
        rows = [_row_scores(pre[0, -1, :], c_ids[0], tok)]
        for j in range(1, len(c_ids)):
            rows.append(_row_scores(cl[0, j - 1, :], c_ids[j], tok))
        results.append(_pack(tok, cont, cont, rows))
    return results


def score_candidates_cached(model, tok, prompt, candidates):
    """Jev-shaped decision call: prefill the state ONCE, then score every
    candidate against the cached prefix. Cost = 1 prefill + N short passes
    instead of N full prefills. Numerically identical to score_continuation()."""
    return score_candidates_cached_ids(model, tok, _encode(tok, prompt, add_bos=True), candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text")
    ap.add_argument("--prompt")
    ap.add_argument("--cont", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--context", default="")
    a = ap.parse_args()

    model, tok = load_model(a.model)

    if a.text is not None:
        ids = _encode(tok, a.text, add_bos=True)
        rows = score_tokens(model, tok, ids)
        if a.json:
            print(json.dumps(rows, indent=1))
            return
        print(f"{'token':>18} {'p':>10} {'rank':>7} {'pct_out':>8} {'mass_abv':>9} {'surpr':>7}")
        for r in rows:
            print(f"{r['token']!r:>18} {r['p']:>10.6f} {r['rank']:>7d} "
                  f"{r['pct_outside']:>7.2f}% {r['mass_above']:>9.6f} {r['surprisal']:>7.3f}")
        return

    if a.prompt and a.cont:
        results = []
        for c in a.cont:
            r = score_continuation(model, tok, a.prompt, c)
            results.append(r)
        if a.json:
            print(json.dumps(results, indent=1))
            return
        print(f"prompt: {a.prompt!r}\n")
        print(f"{'answer':>34} {'n':>3} {'mean_lp':>9} {'ppl':>10} {'p(geo)':>10} {'weakest':>10}")
        for r in sorted(results, key=lambda x: -x["mean_lp"]):
            print(f"{r['continuation']!r:>34} {r['n_tokens']:>3} {r['mean_logprob']:>9.3f} "
                  f"{r['perplexity']:>10.3f} {r['geometric_prob']:>10.4f} "
                  f"{r['worst_token']['p']:>10.5f}")
        return

    print("nothing to do: pass --text or --prompt/--cont", file=sys.stderr)


if __name__ == "__main__":
    main()
