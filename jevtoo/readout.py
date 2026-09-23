"""Core readout: turn any causal LM into a typed decision model.

The whole trick is that you never generate anything. A decoder-only LM with a
causal mask computes, at position t, a distribution over the vocabulary that
predicts token t+1. Feed it ``[state + explicit options]`` and the final
position's distribution already contains the probability of every option.

If each option is written as a single-token label (``" A"``, ``" B"``, ...), then
reading ``softmax(logits[-1])[label_token_id]`` gives that option's probability
in ONE forward pass, with no sampling and no second pass.

This module is backend-agnostic in shape; the shipped backend is MLX (Apple
Silicon). A transformers backend implements the same three calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

LETTERS = [chr(65 + i) for i in range(26)]
DEFAULT_LABEL_STYLE = "spaced"  # " A" rather than "A"
DEFAULT_SUFFIX = "\n\nAnswer:"


def label_token(style: str, i: int) -> str:
    """The literal text whose probability we read for option i."""
    letter = LETTERS[i]
    return f" {letter}" if style == "spaced" else letter


@dataclass
class Distribution:
    """A typed decision, not prose."""

    labels: list[str]
    probs: list[float]
    raw: list[float] = field(default_factory=list)
    per_token: list[list[float]] = field(default_factory=list)  # non-`spaced` labels may span tokens
    latency_s: float | None = None

    # -- typed accessors -------------------------------------------------
    @property
    def share(self) -> list[float]:
        """Option probabilities renormalised over the option set only.

        This is the number to branch on: it answers "given that the answer is
        one of these options, which one?" and removes the mass the model put
        on everything else.
        """
        total = sum(self.raw or self.probs)
        if total <= 0:
            return [0.0] * len(self.probs)
        return [p / total for p in (self.raw or self.probs)]

    @property
    def choice(self) -> str:
        """CHOlCE primitive: argmax label."""
        i = max(range(len(self.probs)), key=lambda k: (self.raw or self.probs)[k])
        return self.labels[i]

    @property
    def confidence(self) -> float:
        return max(self.share) if self.probs else 0.0

    def noul(self, positive: str | None = None) -> float:
        """NOUL primitive: P(yes) for a yes/no question.

        The positive label is resolved by NAME, not by position. Assuming the
        positive label sits at index 1 silently returns P(no) whenever a caller
        passes ["yes", "no"] - which is precisely the option-order bug this
        package exists to warn about.
        """
        if positive is None:
            for candidate in ("yes", "true", "y"):
                if candidate in self.labels:
                    positive = candidate
                    break
            else:
                positive = self.labels[-1]
        return self.share[self.labels.index(positive)]

    def score(self) -> float:
        """SCORE primitive: expected value over ordinal levels 0..n-1."""
        sh = self.share
        return sum(i * s for i, s in enumerate(sh))

    def as_dict(self) -> dict:
        return {
            "labels": self.labels,
            "choice": self.choice,
            "confidence": self.confidence,
            "share": dict(zip(self.labels, self.share)),
            "raw": dict(zip(self.labels, self.raw or self.probs)),
        }


def render_options(labels: Sequence[str], criteria: dict | list | None = None) -> str:
    """Render the option block. Including the rubric text matters: the model is
    choosing between *described* options, not bare identifiers."""
    lines = []
    for i, lab in enumerate(labels):
        desc = ""
        if isinstance(criteria, dict):
            if lab in criteria:
                desc = criteria[lab]
            elif lab in ("yes", "no"):
                desc = criteria.get("true" if lab == "yes" else "false", "")
        elif isinstance(criteria, list) and i < len(criteria):
            c = criteria[i]
            desc = c if isinstance(c, str) else str(c)
        lines.append(f"{LETTERS[i]}. {lab}" + (f" - {desc}" if desc else ""))
    return "\n".join(lines)


def render_state(state: str, instructions: str, labels: Sequence[str],
                 criteria: dict | list | None = None,
                 suffix: str = DEFAULT_SUFFIX) -> str:
    return f"{state}\n\n{instructions}\n\n{render_options(labels, criteria)}{suffix}"


def softmax(xs: Iterable[float]) -> list[float]:
    xs = list(xs)
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def token_diagnostics(logits_row, token_id: int) -> dict:
    """Per-token diagnostics used when the given token is a full string rather
    than a single label. Works on a raw logits vector."""
    probs = softmax(logits_row)
    p = probs[token_id]
    rank = sum(1 for v in logits_row if v > logits_row[token_id])
    return {
        "p": p,
        "rank": rank,
        "pct_outside": 100.0 * (1.0 - p),
        "surprisal": -math.log(max(p, 1e-12)),
        "margin": max(logits_row) - logits_row[token_id],
    }
