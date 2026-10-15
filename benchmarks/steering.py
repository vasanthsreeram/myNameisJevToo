"""Jev steering experiment: a small model generates, a large model steers.

The generator is a cheap model (default Qwen3.5-0.8B) that writes an SVG. The
steerer is a large model served through this repo's typed decision readout: it
never writes prose, it only picks a label from a fixed vocabulary at each
checkpoint. That label is handed back to the generator as steering.

Run:

    python benchmarks/steering.py --prompt "a snake" --rounds 4
    python benchmarks/steering.py --prompt "a snake" --rounds 4 --mode baseline

``--mode baseline`` spends the same generation budget with no steering, so the
two arms are comparable at a matched budget.

Knobs the experiment is meant to sweep:

* ``--rounds``      how many checkpoints (steering passes) per drawing
* ``--vocab``       which steering vocabulary to use (``basic`` or ``full``)
* ``--mode``        ``steer`` / ``baseline`` / ``judge`` (score only, no feedback)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from jevtoo.readout import peakedness

# --------------------------------------------------------------------------
# Steer vocabularies. Labels are single letters in the prompt (" A", " B", ...),
# so each readout is one forward pass regardless of how many labels there are.
# The rubric text is what the model actually decides between -- bare identifiers
# are not enough.
# --------------------------------------------------------------------------

VOCABS = {
    "basic": {
        "labels": ["continue", "fix", "done"],
        "criteria": {
            "continue": "the drawing is on track; keep building it the same way",
            "fix": "something is wrong or crude; correct it before adding more",
            "done": "the drawing is complete and clearly reads as the subject; stop",
        },
    },
    "full": {
        "labels": [
            "continue",
            "fix_geometry",
            "fix_scale",
            "fix_color",
            "add_detail",
            "simplify",
            "redo",
            "done",
        ],
        "criteria": {
            "continue": "the last chunk is on track; keep building the same plan",
            "fix_geometry": "shapes are wrong, misaligned, or not closed",
            "fix_scale": "proportions, viewBox, or sizing is off",
            "fix_color": "fills, strokes, or palette do not match the subject",
            "add_detail": "too plain; missing the features that identify the subject",
            "simplify": "too busy, noisy, or cluttered",
            "redo": "this direction is wrong; rebuild this part from scratch",
            "done": "the drawing is complete and clearly reads as the subject; stop",
        },
    },
}

QUALITY_LEVELS = ["unusable", "rough", "recognisable", "clean", "polished"]

SVG_RE = re.compile(r"<svg[\s\S]*?</svg>", re.I)


def extract_svg(text: str) -> str | None:
    """First complete ``<svg>`` element in the text, or None."""
    m = SVG_RE.search(text or "")
    return m.group(0) if m else None


def partial_svg(text: str) -> str:
    """Renderable snapshot of an SVG still being written.

    Takes the text from ``<svg`` onward and closes the root if it is still open,
    so a half-written element still paints. Returns '' before any ``<svg``.
    """
    s = text or ""
    i = s.lower().find("<svg")
    if i < 0:
        return ""
    s = s[i:]
    if "</svg>" not in s.lower():
        s += "</svg>"
    return s


# --------------------------------------------------------------------------
# Generator: the cheap model doing the actual writing.
# --------------------------------------------------------------------------


class Generator:
    def __init__(self, model_path: str, max_tokens: int = 700):
        from mlx_lm import load

        print(f"[generator] loading {model_path} ...", flush=True)
        t0 = time.time()
        self.model, self.tokenizer = load(model_path)
        self.max_tokens = max_tokens
        print(f"[generator] ready in {time.time() - t0:.1f}s", flush=True)

    def write(self, subject: str, svg_so_far: str, steering: str | None) -> str:
        """One generation round. Returns the model's raw text."""
        from mlx_lm import generate

        parts = [
            "Draw a simple SVG picture. Output the <svg> code only.",
            f'Subject: "{subject}".',
        ]
        if svg_so_far:
            parts.append("Current SVG:\n" + svg_so_far)
            if steering:
                parts.append(f"Feedback: {steering}")
                parts.append("Fix it and output the whole corrected SVG.")
            else:
                parts.append("Keep drawing. Output the whole SVG.")
        else:
            parts.append("Start now.")
        prompt = "\n\n".join(parts)

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )


# --------------------------------------------------------------------------
# Steerer: the large model making typed decisions at each checkpoint.
# --------------------------------------------------------------------------


class Steerer:
    def __init__(self, model_path: str, vocab_name: str):
        from jevtoo import convert

        print(f"[steerer] loading {model_path} ...", flush=True)
        t0 = time.time()
        self.jev = convert(model_path)
        self.vocab = VOCABS[vocab_name]
        self.vocab_name = vocab_name
        print(f"[steerer] ready in {time.time() - t0:.1f}s", flush=True)

    def _state(self, subject: str, svg: str, chunk: str, round_no: int) -> str:
        return (
            f"A small model is drawing an SVG of \"{subject}\".\n"
            f"Checkpoint {round_no}.\n\n"
            f"--- SVG so far ---\n{svg or '(nothing yet)'}\n\n"
            f"--- newest chunk ---\n{chunk or '(nothing yet)'}"
        )

    def judge(self, subject: str, svg: str, chunk: str, round_no: int) -> dict:
        """One checkpoint: steering label, stop gate, and quality score."""
        state = self._state(subject, svg, chunk, round_no)

        labels = self.vocab["labels"]
        d = self.jev.choice(
            state,
            "Which steering action should the generator take next?",
            labels,
            self.vocab["criteria"],
        )

        # Binary gate. The option order is swapped on purpose: a two-label read
        # can reflect the first slot's position more than the state, so the
        # answer is only trusted when the reversed order agrees.
        p_stop = self.jev.noul(
            state, "Is this drawing complete and good enough to stop?", yes="yes", no="no"
        )
        d_rev = self.jev.options(
            state,
            ["yes", "no"],
            "Is this drawing complete and good enough to stop?",
            {"yes": "finished; stop now", "no": "keep going"},
        )
        p_stop_rev = d_rev.noul(positive="yes")

        qual, d_qual = self.jev.score(
            state, "How good is this drawing so far?", QUALITY_LEVELS
        )

        return {
            "round": round_no,
            "steering": {
                "choice": d.choice,
                "confidence": d.confidence,
                "share": dict(zip(d.labels, d.share)),
                "raw": dict(zip(d.labels, d.raw)),
                "option_mass": sum(d.raw),
                "peakedness": peakedness(d.share),
            },
            "stop": {
                "p_yes": p_stop,
                "p_yes_reversed": p_stop_rev,
                "agree": abs(p_stop - p_stop_rev) < 0.25,
                "mean": (p_stop + p_stop_rev) / 2.0,
            },
            "quality": {"expected": qual, "share": dict(zip(d_qual.labels, d_qual.share))},
        }


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------


def run(args) -> dict:
    gen = Generator(args.generator, max_tokens=args.max_tokens)
    steer = (
        Steerer(args.steerer, args.vocab)
        if args.mode in ("steer", "judge")
        else None
    )

    subject = args.prompt
    svg = ""
    chunk = ""
    log = []

    for r in range(1, args.rounds + 1):
        print(f"\n=== round {r}/{args.rounds} ===", flush=True)

        feedback = None
        verdict = None
        if steer is not None and r > 1:
            verdict = steer.judge(subject, svg, chunk, r)
            verdict.pop("round", None)
            log.append({"round": r, "kind": "judge", **verdict})

            if args.mode == "steer":
                choice = verdict["steering"]["choice"]
                if choice == "done" and verdict["stop"]["agree"] and verdict["stop"]["mean"] > 0.6:
                    print("[loop] steerer says done -- stopping early", flush=True)
                    break
                feedback = _feedback_line(verdict)
                print(f"[steer] {feedback}", flush=True)

        t0 = time.time()
        raw = gen.write(subject, svg, feedback)
        dt = time.time() - t0

        new_svg = extract_svg(raw) or partial_svg(raw)
        if new_svg:
            if svg and new_svg.strip() == svg.strip():
                pass
            else:
                chunk = new_svg if not svg else _diff_hint(svg, new_svg)
                svg = new_svg
        else:
            chunk = ""
            print("[generator] no SVG in output; keeping previous", flush=True)

        log.append(
            {
                "round": r,
                "kind": "generate",
                "seconds": round(dt, 2),
                "mode": args.mode,
                "feedback": feedback,
                "svg_chars": len(svg),
                "raw_chars": len(raw),
            }
        )
        print(f"[gen] {dt:.1f}s  svg {len(svg)} chars", flush=True)

    return {
        "prompt": subject,
        "mode": args.mode,
        "vocab": args.vocab,
        "rounds": args.rounds,
        "generator": args.generator,
        "steerer": args.steerer if steer is not None else None,
        "svg": svg,
        "log": log,
    }


def _feedback_line(verdict: dict) -> str:
    s = verdict["steering"]
    q = verdict["quality"]["expected"]
    return f"{s['choice']} (conf {s['confidence']:.2f}, quality {q:.2f}/4)"


def _diff_hint(old: str, new: str) -> str:
    """Cheap stand-in for 'the newest chunk': how much of the SVG changed."""
    if len(new) > len(old):
        return f"added ~{len(new) - len(old)} characters"
    return f"rewrote the SVG ({len(old)} -> {len(new)} characters)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="a snake", help="what to draw")
    ap.add_argument("--rounds", type=int, default=4, help="checkpoints per drawing")
    ap.add_argument("--mode", choices=["steer", "baseline", "judge"], default="steer")
    ap.add_argument("--vocab", choices=sorted(VOCABS), default="full")
    ap.add_argument("--generator", default="mlx-community/Qwen3.5-0.8B-8bit")
    ap.add_argument("--steerer", default="lmstudio-community/Qwen3.8-27B-MLX-4bit")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--out", default=None, help="where to write the result JSON")
    args = ap.parse_args()

    result = run(args)

    out = Path(args.out) if args.out else Path("benchmarks/results") / (
        f"steering_{args.mode}_{args.vocab}_{int(time.time())}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {out}")

    svg_path = out.with_suffix(".svg")
    if result["svg"]:
        svg_path.write_text(result["svg"])
        print(f"[saved] {svg_path}")


if __name__ == "__main__":
    main()
