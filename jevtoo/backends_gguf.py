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

import http.client
import json
import math
import threading
import time
import urllib.parse

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
        self.stats = {
            "decisions": 0, "labels_missing": 0, "labels_total": 0, "latency_s": 0.0,
        }
        self._http_lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None
        self._reset_conn()
        self.info = self._get("/props")

    # -- http helpers ----------------------------------------------------
    def _reset_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        parsed = urllib.parse.urlparse(self.base)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        host = parsed.hostname or "127.0.0.1"
        if parsed.scheme == "https":
            self._conn = http.client.HTTPSConnection(host, port, timeout=self.timeout)
        else:
            self._conn = http.client.HTTPConnection(host, port, timeout=self.timeout)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """One keep-alive connection for tokenize and completion.

        urllib opened a new TCP connection per call, and a decision pays for a
        tokenize per new label plus the completion. The body is read in full
        before the connection is reused.
        """
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Connection": "keep-alive"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        last_exc: Exception | None = None
        for _attempt in (1, 2):
            try:
                with self._http_lock:
                    assert self._conn is not None
                    self._conn.request(method, path, body=body, headers=headers)
                    resp = self._conn.getresponse()
                    data = resp.read()
                    status = resp.status
            except Exception as exc:  # noqa: BLE001 - reconnect once, then raise
                last_exc = exc
                self._reset_conn()
                continue
            if status >= 400:
                raise RuntimeError(f"llama-server {path} returned {status}: {data[:200]!r}")
            return json.loads(data)
        assert last_exc is not None
        raise last_exc

    def _post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, payload)

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

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
        self.stats["latency_s"] += latency
        evaluated = resp.get("tokens_evaluated")
        cached = resp.get("tokens_cached")
        return Distribution(
            labels=labels, probs=raw, raw=raw, latency_s=latency, missing=missing,
            n_tokens=evaluated if isinstance(evaluated, int) else None,
            cached_tokens=cached if isinstance(cached, int) else None,
        )

    def coverage(self) -> float:
        t = self.stats["labels_total"]
        return 1.0 - (self.stats["labels_missing"] / t) if t else 1.0
