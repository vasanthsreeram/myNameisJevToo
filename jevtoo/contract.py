"""The TypeSafe ``POST /v1/systemone`` shape, on top of a local readout.

The request and the answer fields follow the published API:
https://docs.typesafe.ai/api

What their response leaves out — queue time, how much probability sat on the
option letters before renormalising, the winner's share beside peakedness —
is returned only when the caller asks, under ``observability``. The default
body stays the published three keys: ``model``, ``answers``, ``usage``.
"""
from __future__ import annotations

import json
from typing import Any

from .readout import peakedness, render_state

MAX_CHOICE = 26  # one single-token letter per option, A through Z
MIN_SCORE = 2
MAX_SCORE = 10


class ContractError(Exception):
    def __init__(self, message: str, field: str | None = None, status: int = 422):
        super().__init__(message)
        self.message = message
        self.field = field
        self.status = status


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _require_text(value: Any, field: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ContractError(f"{field} is empty", field)
        return value
    if isinstance(value, (dict, list)):
        return as_text(value)
    raise ContractError(f"{field} must be a string, object, or array", field)


def _round_share(labels: list[str], share: list[float]) -> dict[str, float]:
    """Six decimals, then put the rounding drift on the winner so the values sum to 1."""
    vals = [round(float(s), 6) for s in share]
    if not vals:
        return {}
    drift = round(1.0 - sum(vals), 6)
    winner = max(range(len(vals)), key=lambda i: vals[i])
    vals[winner] = round(vals[winner] + drift, 6)
    return dict(zip(labels, vals))


def _check_model(requested: Any, model_id: str, aliases: set[str]) -> None:
    if not isinstance(requested, str) or not requested.strip():
        raise ContractError("model is required", "model")
    if requested != model_id and requested not in aliases:
        known = ", ".join(sorted({model_id, *aliases}))
        raise ContractError(
            f"model {requested!r} is not loaded. This server is serving {model_id!r} ({known}).",
            "model",
        )


def _question_prompt(state: str, question: dict, field: str) -> tuple[str, list[str], str]:
    """Returns ``(prompt, labels, kind)``."""
    if not isinstance(question, dict):
        raise ContractError(f"{field} must be an object", field)
    kind = question.get("type")
    if kind not in ("choice", "noul", "score"):
        raise ContractError(f"{field}.type must be choice, noul, or score", f"{field}.type")
    instructions = _require_text(question.get("instructions"), f"{field}.instructions")
    criteria = question.get("criteria", None)

    if kind == "noul":
        labels = ["no", "yes"]
        rendered = None
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise ContractError(f"{field}.criteria must be an object", f"{field}.criteria")
            rendered = {}
            for key in ("true", "false"):
                if key in criteria and criteria[key] is not None:
                    value = criteria[key]
                    rendered[key] = value if isinstance(value, str) else as_text(value)
        return render_state(state, instructions, labels, rendered), labels, kind

    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ContractError(
                f"{field}.criteria must be a map of option to description",
                f"{field}.criteria",
            )
        if len(criteria) > MAX_CHOICE:
            raise ContractError(
                f"{field} has {len(criteria)} options. This readout uses one single-token "
                f"letter per option, so the limit is {MAX_CHOICE}.",
                f"{field}.criteria",
            )
        labels = [str(key) for key in criteria]
        rendered = {}
        for key, value in criteria.items():
            if value is None:
                continue
            rendered[str(key)] = value if isinstance(value, str) else as_text(value)
        return render_state(state, instructions, labels, rendered or None), labels, kind

    if not isinstance(criteria, list) or not (MIN_SCORE <= len(criteria) <= MAX_SCORE):
        raise ContractError(
            f"{field}.criteria must be an ordered list of {MIN_SCORE} to {MAX_SCORE} levels",
            f"{field}.criteria",
        )
    labels = []
    for i, level in enumerate(criteria):
        text = level if isinstance(level, str) else as_text(level)
        if not str(text).strip():
            raise ContractError(f"{field}.criteria[{i}] is empty", f"{field}.criteria")
        labels.append(str(text))
    if len(set(labels)) != len(labels):
        raise ContractError(f"{field}.criteria levels must be distinct", f"{field}.criteria")
    return render_state(state, instructions, labels, None), labels, "score"


def _token_count(backend: object, prompt: str, dist) -> int:
    if getattr(dist, "n_tokens", None) is not None:
        return int(dist.n_tokens)
    encode = getattr(backend, "encode", None)
    if not callable(encode):
        return 0
    try:
        return len(encode(prompt, add_bos=True))
    except TypeError:
        return len(encode(prompt))


def _answer(kind: str, labels: list[str], dist) -> dict:
    share = dist.share
    if kind == "noul":
        return {"type": "noul", "noul": round(float(dist.noul(positive="yes")), 6)}
    confidence = round(float(peakedness(share)), 6)
    if kind == "choice":
        return {
            "type": "choice",
            "choice": dist.choice,
            "probabilities": _round_share(labels, share),
            "confidence": confidence,
        }
    return {
        "type": "score",
        "score": round(float(dist.score()), 6),
        "legend": {str(i): lab for i, lab in enumerate(labels)},
        "probabilities": _round_share([str(i) for i in range(len(labels))], share),
        "confidence": confidence,
    }


def _question_obs(dist, labels: list[str], prompt_tokens: int) -> dict:
    share = dist.share
    obs = {
        "forward_ms": round(float(dist.latency_s or 0.0) * 1000, 3),
        "input_tokens": prompt_tokens,
        "option_mass": round(float(sum(dist.raw or dist.probs)), 6),
        "winner_probability": round(float(max(share) if share else 0.0), 6),
        "peakedness": round(float(peakedness(share)), 6),
        "missing_labels": int(getattr(dist, "missing", 0) or 0),
    }
    if getattr(dist, "cached_tokens", None) is not None:
        obs["cached_tokens"] = int(dist.cached_tokens)
    if len(labels) == 2:
        obs["binary_position_prior"] = True
    return obs


def evaluate(model, body: Any, *, model_id: str, aliases: set[str], observe: bool) -> tuple[dict, dict]:
    """Validate one request and read every question.

    Returns ``(response, observability)``. ``response`` includes the
    observability object only when ``observe`` is set.
    """
    if not isinstance(body, dict):
        raise ContractError("request body must be a JSON object")
    _check_model(body.get("model"), model_id, aliases)
    if "state" not in body or body["state"] is None:
        raise ContractError("state is required", "state")
    state = body["state"] if isinstance(body["state"], str) else as_text(body["state"])
    questions = body.get("questions")
    if not isinstance(questions, dict):
        raise ContractError("questions must be an object", "questions")

    prepared: list[tuple[str, str, list[str], str]] = []
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            raise ContractError("question ids must be non-empty strings", "questions")
        prompt, labels, kind = _question_prompt(state, question, f"questions.{qid}")
        prepared.append((qid, kind, labels, prompt))

    items = [(prompt, labels) for _qid, _kind, labels, prompt in prepared]
    dists = model.distributions(items) if items else []
    dists = _retry_partial_gguf(model, items, dists)

    answers: dict[str, dict] = {}
    per_question: dict[str, dict] = {}
    input_tokens = 0
    for (qid, kind, labels, prompt), dist in zip(prepared, dists):
        answers[qid] = _answer(kind, labels, dist)
        count = _token_count(model.backend, prompt, dist)
        input_tokens += count
        per_question[qid] = _question_obs(dist, labels, count)

    prefix = dists[0].prefix_tokens if dists and dists[0].prefix_tokens else 0
    if prefix:
        computed = prefix + sum(max(0, (d.n_tokens or 0) - prefix) for d in dists)
        prefill_ms = round(max(d.prefill_s or 0.0 for d in dists) * 1000, 3)
    else:
        computed = input_tokens
        prefill_ms = 0.0
    forward_ms = round(sum(float(d.latency_s or 0.0) for d in dists) * 1000, 3)
    coverage = None
    cov = getattr(model.backend, "coverage", None)
    if callable(cov):
        coverage = cov()

    obs = {
        "model_id": model_id,
        "questions": per_question,
        "shared_prefix_tokens": prefix,
        "prefill_ms": prefill_ms,
        "forward_ms": forward_ms,
        "computed_tokens": computed,
        "coverage": coverage,
        "head_mode": getattr(model.backend, "_head_mode", None)
        or (getattr(model.backend, "stats", {}) or {}).get("head_mode"),
    }
    response = {
        "model": model_id,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": len(answers)},
    }
    if observe:
        response["observability"] = obs
    return response, obs


def _retry_partial_gguf(model, items, dists):
    """A GGUF label missing from top-N is a silent 0. Widen the window once."""
    backend = model.backend
    n_probs = getattr(backend, "n_probs", None)
    if not isinstance(n_probs, int) or not dists:
        return dists
    if not any(int(getattr(d, "missing", 0) or 0) > 0 for d in dists):
        return dists
    widened = min(max(n_probs * 5, 1000), 5000)
    if widened <= n_probs:
        return dists
    backend.n_probs = widened
    try:
        return model.distributions(items)
    finally:
        backend.n_probs = n_probs
