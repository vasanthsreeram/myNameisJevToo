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

# Below this, cloning or trimming a KV cache beats re-prefilling the state.
SHARED_PREFIX_MIN = 32


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Longest token prefix shared by every sequence.

    Compare encoded prompts, not concatenated strings. BPE can merge across a
    boundary, and a prefix that only exists in text is not a prefix in ids.
    """
    if not seqs:
        return 0
    limit = min(len(seq) for seq in seqs)
    first = seqs[0]
    n = 0
    while n < limit:
        tok = first[n]
        if any(seq[n] != tok for seq in seqs[1:]):
            break
        n += 1
    return n


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
        # "last" once a last-token lm_head matches a full forward. Until then,
        # and whenever it does not match, every read is a full forward.
        self._head_mode: str | None = None
        self.stats = {
            "decisions": 0, "labels_missing": 0, "labels_total": 0,
            "latency_s": 0.0, "head_mode": "unprobed",
        }

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

    def _logits_to_raw(self, last, n_labels: int) -> list[float]:
        """P(label) from the final-position logits.

        ``exp(logit_i - logsumexp)`` is the softmax probability of token i.
        Gathering the label ids avoids a vocab-sized softmax and N GPU syncs.
        """
        mx = self.mx
        last = last.astype(mx.float32)
        ids = mx.array([self.label_id(i) for i in range(n_labels)])
        chosen = mx.take(last, ids)
        raw = mx.exp(chosen - mx.logsumexp(last, axis=0))
        mx.eval(raw)
        return [float(x) for x in raw.tolist()]

    def _call_model(self, token_ids: list[int], cache=None):
        x = self.mx.array([token_ids])
        if cache is None:
            return self.model(x)
        return self.model(x, cache=cache)

    def _project_last(self, token_ids: list[int], cache=None):
        """Run the transformer over the tokens and the head on the last one.

        A full ``model()`` builds logits for every position. At a large vocab
        that tensor dominates the read. This is only used after `_probe` has
        shown it matches the full forward on this checkpoint.
        """
        mx = self.mx
        model = self.model
        body = getattr(model, "model", None)
        if body is None:
            raise RuntimeError("model has no transformer body")
        x = mx.array([token_ids])
        hidden = body(x) if cache is None else body(x, cache=cache)
        last_h = hidden[:, -1:, :]
        args = getattr(model, "args", None)
        tied = bool(args is not None and getattr(args, "tie_word_embeddings", False))
        if tied:
            embed = getattr(body, "embed_tokens", None)
            if embed is None or not hasattr(embed, "as_linear"):
                raise RuntimeError("tied embeddings have no as_linear")
            row = embed.as_linear(last_h)
        else:
            head = getattr(model, "lm_head", None)
            if head is None:
                raise RuntimeError("model has no lm_head")
            row = head(last_h)
        return row[0, -1, :]

    def _probe(self, token_ids: list[int]):
        """One full forward, compared with the last-token head. The full row is
        the result, so the probe is not an extra read on the first call."""
        full = self._call_model(token_ids)[0, -1, :]
        try:
            fast = self._project_last(token_ids)
            delta_a = self.mx.max(self.mx.abs(full.astype(self.mx.float32) - fast.astype(self.mx.float32)))
            self.mx.eval(full, fast, delta_a)
            delta = float(delta_a.item()) if hasattr(delta_a, "item") else float(delta_a)
            self._head_mode = "last" if delta < 1e-3 else "full"
            self.stats["head_delta"] = delta
        except Exception as exc:  # noqa: BLE001 - a failed probe means "use the full forward"
            self._head_mode = "full"
            self.stats["head_error"] = type(exc).__name__
        self.stats["head_mode"] = self._head_mode
        return full

    def _last_row(self, token_ids: list[int], cache=None):
        if self._head_mode is None:
            if cache is not None:
                self._probe(token_ids)
            else:
                return self._probe(token_ids)
        if self._head_mode == "last":
            return self._project_last(token_ids, cache)
        return self._call_model(token_ids, cache)[0, -1, :]

    def _record(self, labels: list[str], raw: list[float], latency: float, n_tokens: int,
                prefill_s: float | None = None, prefix_tokens: int | None = None) -> Distribution:
        self.stats["decisions"] += 1
        self.stats["labels_total"] += len(labels)
        self.stats["latency_s"] += latency
        return Distribution(
            labels=list(labels), probs=raw, raw=raw, latency_s=latency,
            n_tokens=n_tokens, prefill_s=prefill_s, prefix_tokens=prefix_tokens,
        )

    def distribution(self, prompt: str, labels: list[str]) -> Distribution:
        """One forward pass over `prompt`; read P(label) for each option."""
        for i in range(len(labels)):
            self.label_id(i)
        ids = self.encode(prompt, add_bos=True)
        t0 = time.perf_counter()
        raw = self._logits_to_raw(self._last_row(ids), len(labels))
        latency = time.perf_counter() - t0
        return self._record(labels, raw, latency, len(ids))

    def distributions(self, items: list[tuple[str, list[str]]]) -> list[Distribution]:
        """Score several prompts. Share one prefill when the token prefix is long.

        Each item is ``(prompt, labels)``. The prefix is the longest common
        *token* prefix. Under 32 tokens, or a prompt that is only the prefix,
        each item is its own forward.
        """
        if len(items) <= 1:
            return [self.distribution(*items[0])] if items else []
        for prompt, labels in items:
            for i in range(len(labels)):
                self.label_id(i)
        encoded = [self.encode(prompt, add_bos=True) for prompt, _ in items]
        lcp = common_prefix_len(encoded)
        if lcp < SHARED_PREFIX_MIN or any(len(ids) <= lcp for ids in encoded):
            return [self.distribution(prompt, labels) for prompt, labels in items]
        try:
            return self._distributions_shared(items, encoded, lcp)
        except Exception as exc:  # noqa: BLE001 - a broken cache must not change the answers
            import sys
            print(
                f"jevtoo: shared-prefix forward failed ({type(exc).__name__}: {exc}); "
                "scoring each question on its own",
                file=sys.stderr,
            )
            return [self.distribution(prompt, labels) for prompt, labels in items]

    def _realize_cache(self, cache) -> None:
        mx = self.mx
        for layer in cache:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                mx.eval(keys, values)

    def _distributions_shared(self, items, encoded, lcp: int) -> list[Distribution]:
        from mlx_lm.models.cache import make_prompt_cache

        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        except ImportError:
            can_trim_prompt_cache = lambda _cache: False  # noqa: E731
            trim_prompt_cache = None

        prefix = encoded[0][:lcp]
        cache = make_prompt_cache(self.model)
        t_pre = time.perf_counter()
        self._last_row(prefix, cache=cache)
        self._realize_cache(cache)
        prefill_s = time.perf_counter() - t_pre
        self.stats["latency_s"] += prefill_s
        try:
            trimmable = bool(can_trim_prompt_cache(cache)) and trim_prompt_cache is not None
        except Exception:  # noqa: BLE001
            trimmable = False

        out = []
        for (_prompt, labels), ids in zip(items, encoded):
            suffix = ids[lcp:]
            cache_i = cache if trimmable else copy.deepcopy(cache)
            t0 = time.perf_counter()
            raw = self._logits_to_raw(self._last_row(suffix, cache=cache_i), len(labels))
            latency = time.perf_counter() - t0
            if trimmable:
                trim_prompt_cache(cache, len(suffix))
            out.append(self._record(
                labels, raw, latency, len(ids), prefill_s=prefill_s, prefix_tokens=lcp,
            ))
        return out

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
