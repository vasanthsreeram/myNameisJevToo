"""Backends. The shipped one is MLX (Apple Silicon); the interface is three calls.

Anything that can give you the final-position logits of a causal LM can back this
package. To add a backend, implement `encode` and `distribution`.

The two things a backend MUST get right:

1. BOS exactly once. ``mlx_lm``'s TokenizerWrapper.encode() prepends the BOS token
   on EVERY call. Concatenating ``encode(prompt) + encode(option)`` therefore
   injects a bogus ``<s>`` between them, which destroys the option's probability.
   On MiniCPM5-2B that single mistake took measured accuracy from 96% to 40% while
   still producing plausible-looking output. Always encode with
   ``add_special_tokens=False`` and prepend BOS yourself, once, at the front.

2. Prefill once, reuse the prefix. Scoring N options by re-running the whole state
   N times is N-1 wasted prefills. Clone the KV cache instead: 4.2x faster and
   bit-identical (max |d mean logprob| = 0.00000000 nats on MiniCPM5-2B).
"""
from __future__ import annotations

import copy
import time

from .readout import Distribution, label_token


class MLXBackend:
    """Causal LM on Apple Silicon via mlx-lm."""

    def __init__(self, model_path: str, label_style: str = "spaced"):
        import mlx.core as mx  # noqa: F401  (import check happens here)
        from mlx_lm import load

        self.mx = mx
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path
        self.label_style = label_style
        self._label_ids: dict[int, int] = {}
        self.stats = {"decisions": 0, "labels_missing": 0, "labels_total": 0}

    # -- tokenisation ----------------------------------------------------
    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Not every tokenizer defines a BOS. Qwen3.5's is None, and blindly
        # prepending it yields [None] + ids, which throws at mx.array() time -
        # and the exception surfaces once per item, so the whole run reads as
        # "0 items scored" rather than as a tokenizer problem.
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if add_bos and bos is not None:
            ids = [bos] + ids
        return ids

    def label_id(self, i: int) -> int:
        """Token id for option i's label. Cached; asserts it is a single token,
        because the single-token property is what makes the read free."""
        if i not in self._label_ids:
            ids = self.encode(label_token(self.label_style, i))
            if len(ids) != 1:
                raise ValueError(
                    f"label {i} tokenised to {len(ids)} tokens ({ids}); "
                    "the readout needs a single-token label"
                )
            self._label_ids[i] = ids[0]
        return self._label_ids[i]

    # -- the read --------------------------------------------------------
    def coverage(self) -> float:
        """Fraction of labels found in the readout.

        Always 1.0 here: labels are indexed straight into the full softmax over
        the vocabulary, so unlike the GGUF top-N path there is nothing to miss.
        Exposed anyway so both backends report the same summary fields.
        """
        return 1.0

    def distribution(self, prompt: str, labels: list[str]) -> Distribution:
        """One forward pass over `prompt`; read P(label) for each option."""
        mx = self.mx
        ids = self.encode(prompt, add_bos=True)
        t0 = time.perf_counter()
        logits = self.model(mx.array([ids]))
        mx.eval(logits)
        probs = mx.softmax(logits[0, -1, :].astype(mx.float32))
        mx.eval(probs)
        latency = time.perf_counter() - t0
        raw = [float(probs[self.label_id(i)]) for i in range(len(labels))]
        self.stats["decisions"] += 1
        self.stats["labels_total"] += len(labels)
        return Distribution(labels=list(labels), probs=raw, raw=raw, latency_s=latency)

    # -- multi-token candidates (slower path, still one prefill) ---------
    def score_candidates(self, prompt: str, candidates: list[str]) -> list[dict]:
        """Score full candidate *strings* instead of single-token labels.

        Use when options are long free text and you want the probability of the
        text itself. Costs one prefill plus one short pass per candidate, against
        the shared cached prefix. Numerically identical to scoring each candidate
        independently.
        """
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache

        p_ids = self.encode(prompt, add_bos=True)
        cache = make_prompt_cache(self.model)
        pre = self.model(mx.array([p_ids]), cache=cache)
        mx.eval(pre)
        for c in cache:
            if getattr(c, "keys", None) is not None:
                mx.eval(c.keys, c.values)

        out = []
        for text in candidates:
            c_ids = self.encode(text)
            if not c_ids:
                continue
            cache_i = copy.deepcopy(cache)
            cl = self.model(mx.array([c_ids]), cache=cache_i)
            mx.eval(cl)
            rows = []
            for j, tid in enumerate(c_ids):
                row = pre[0, -1, :] if j == 0 else cl[0, j - 1, :]
                p = float(mx.softmax(row.astype(mx.float32))[tid])
                rows.append({"token": self.tokenizer.decode([tid]), "p": p,
                             "logprob": __import__("math").log(max(p, 1e-12))})
            total = sum(r["logprob"] for r in rows)
            out.append({
                "candidate": text, "n_tokens": len(rows),
                "total_logprob": total, "mean_logprob": total / len(rows),
                "geometric_prob": __import__("math").exp(total / len(rows)),
                "tokens": rows,
            })
        return out


def load_backend(model_path: str, label_style: str = "spaced"):
    return MLXBackend(model_path, label_style=label_style)
