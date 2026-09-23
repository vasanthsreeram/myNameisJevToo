#!/usr/bin/env python3
"""
mcpm_choice.py - spec-decode style choice scoring, NO text generation.

Contract (what Vas asked for):
  * input  = one state string that already contains the choices (A/B/C/D)
  * model  = MiniCPM5-2B-MLX, 4-bit
  * output = the TOKEN PROBABILITY of each choice, not a generated answer

Mechanism:
  One forward pass over the prefilled state. If a choice label is a single
  token (e.g. " A"), its probability is simply softmax(logits[-1])[that token] -
  no second pass at all. The "spec decode" read is free because the label is
  one token. Multi-token labels fall back to a cached candidate pass.

  Nothing is generated. We only read the distribution.
"""
import math
import sys

import mlx.core as mx
from mlx_lm import load

from _shared_ import _encode, score_candidates_cached_ids  # noqa: E402

LABELS = (" A", " B", " C", " D")


def prefilled_probs(model, tok, state):
    """Prefill the state once. Returns (last-position softmax probs, token ids)."""
    ids = _encode(tok, state, add_bos=True)
    logits = model(mx.array([ids]))
    mx.eval(logits)
    probs = mx.softmax(logits[0, -1, :].astype(mx.float32))
    mx.eval(probs)
    return probs, ids


def choice_probs(model, tok, state, labels=LABELS, normalize=True):
    """Read the probability of each single-token choice label off ONE prefill.

    Returns (per-label dict, normalised share over the label set).
    """
    probs, ids = prefilled_probs(model, tok, state)
    out, multi = {}, []
    for lab in labels:
        l_ids = _encode(tok, lab)
        if len(l_ids) == 1:
            out[lab] = {"token_ids": l_ids, "single_token": True, "p": float(probs[l_ids[0]])}
        else:
            out[lab] = {"token_ids": l_ids, "single_token": False, "p": None}
            multi.append(lab)

    if multi:
        # rare: label split across tokens -> one cached pass per such label
        cached = score_candidates_cached_ids(model, tok, ids, multi)
        for lab, r in zip(multi, cached):
            out[lab]["p"] = r["geometric_prob"]
            out[lab]["per_token"] = [t["p"] for t in r["tokens"]]

    total = sum(v["p"] or 0.0 for v in out.values())
    share = {k: ((v["p"] or 0.0) / total if total > 0 else 0.0) for k, v in out.items()} if normalize else {}
    return out, share


def render_mcq(question, options, answer_prefix="\n\nAnswer:"):
    """options = list of strings in display order (index 0 -> 'A')."""
    body = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    return f"{question}\n\n{body}{answer_prefix}"


def decide(model, tok, question, options, labels=LABELS, verbose=False):
    state = render_mcq(question, options)
    per, share = choice_probs(model, tok, state, labels)
    best = max(per, key=lambda k: per[k]["p"] or 0.0)
    idx = labels.index(best)
    if verbose:
        for lab in labels:
            p = per[lab]["p"]
            print(f"    {lab!r:>5} p={p:.6f}  share={share[lab]*100:6.2f}%  toks={per[lab]['token_ids']}")
        print(f"    -> picks {best!r} = {options[idx]!r}")
    return {"state": state, "per_label": per, "share": share,
            "picked_label": best, "picked_index": idx, "picked": options[idx]}


if __name__ == "__main__":
    model, tok = load("/Users/admin/models/MiniCPM5-2B-MLX")
    print("label tokenisation check:")
    for lab in LABELS:
        ids = _encode(tok, lab)
        print(f"  {lab!r:>5} -> ids={ids} ({len(ids)} token{'s' if len(ids)!=1 else ''}) {tok.decode(ids)!r}")
    print()
    q = "The capital of Australia is"
    opts = ["Canberra", "Sydney", "Melbourne", "Perth"]
    print("state sent to the model:")
    print(render_mcq(q, opts))
    print("\nprobability readout:")
    decide(model, tok, q, opts, verbose=True)
