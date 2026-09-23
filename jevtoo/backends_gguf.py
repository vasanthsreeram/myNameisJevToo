"""GGUF backend: read label probabilities out of a llama.cpp server.

llama.cpp does not hand out full logits over HTTP, but `llama-server`'s
/completion endpoint returns the top-N token probabilities at the next position
via `n_probs`. For multiple-choice decisions the label tokens (" A".." I") sit
comfortably inside the top few hundred, so reading them from the returned list
is equivalent to reading the distribution directly - as long as you check that
the label was actually present.

This backend therefore reports `coverage`: the fraction of decisions where every
label token appeared in the returned top-N. If coverage is low, raise n_probs
before trusting the numbers.

    llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf -c 8192 --port 8080
    jev = convert_gguf("http://127.0.0.1:8080")
"""
from __future__ import annotations

import json
import math
import time
import urllib.request

from .readout import Distribution, label_token


class LlamaCppServerBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 label_style: str = "spaced", n_probs: int = 200, timeout: int = 600):
        self.base = base_url.rstrip("/")
        self.label_style = label_style
        self.n_probs = n_probs
        self.timeout = timeout
        self._label_ids: dict[int, int] = {}
        self._cache: dict[str, int] = {}
        self.stats = {"decisions": 0, "labels_missing": 0, "labels_total": 0}
        self.info = self._get("/props")

    # -- http helpers ----------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return json.loads(r.read())

    def _tokenize(self, text: str) -> list[int]:
        out = self._post("/tokenize", {"content": text})
        return out.get("tokens", [])

    def label_id(self, i: int) -> int:
        """Token id of option i's label, with the BOS pitfall handled by
        tokenising the label with add_special=False semantics (no BOS is ever
        added inside the prompt)."""
        if i not in self._label_ids:
            ids = self._tokenize(label_token(self.label_style, i))
            if len(ids) != 1:
                raise ValueError(f"label {i} tokenised to {len(ids)} tokens ({ids})")
            self._label_ids[i] = ids[0]
        return self._label_ids[i]

    # -- the read --------------------------------------------------------
    def distribution(self, prompt: str, labels: list[str]) -> Distribution:
        n = len(labels)
        labels = list(labels)
        label_ids = [self.label_id(i) for i in range(n)]

        t0 = time.perf_counter()
        resp = self._post("/completion", {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "top_k": 0,
            "n_probs": self.n_probs,
            "post_sampling_probs": False,
            "cache_prompt": True,
            "seed": 0,
        })
        latency = time.perf_counter() - t0

        cp = (resp.get("completion_probabilities") or [{}])[0]
        # llama.cpp renamed this field. Older builds return
        #   completion_probabilities[0]["probs"] -> [{"id", "prob"}]
        # build 11120+ returns
        #   completion_probabilities[0]["top_logprobs"] -> [{"id","token","bytes","logprob"}]
        # Reading only "probs" against a newer server yields an empty list, every
        # label scores 0.0, and the argmax silently falls back to the first
        # option - so support both shapes and never assume one.
        entries = cp.get("top_logprobs") or cp.get("probs") or []
        top: dict[int, float] = {}
        for e in entries:
            tid = e.get("id")
            if tid is None:
                continue
            if "logprob" in e and e["logprob"] is not None:
                top[int(tid)] = math.exp(float(e["logprob"]))
            elif "prob" in e and e["prob"] is not None:
                top[int(tid)] = float(e["prob"])

        raw, missing = [], 0
        for tid in label_ids:
            if tid in top:
                raw.append(top[tid])
            else:
                raw.append(0.0)
                missing += 1
        self.stats["decisions"] += 1
        self.stats["labels_missing"] += missing
        self.stats["labels_total"] += n
        return Distribution(labels=labels, probs=raw, raw=raw, latency_s=latency)

    def coverage(self) -> float:
        t = self.stats["labels_total"]
        return 1.0 - (self.stats["labels_missing"] / t) if t else 1.0
