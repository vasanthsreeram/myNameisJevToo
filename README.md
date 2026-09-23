# myNameisJevToo

Convert **any** causal language model into a Jev-style decision model — no retraining, no
fine-tune, no new weights. Then find out, honestly, whether it is any good.

```python
from jevtoo import convert

jev = convert("openbmb/MiniCPM5-2B-MLX")          # any decoder-only LM

jev.noul(state,   "Has the customer been charged twice this month?")     # -> 0.83
jev.choice(state, "Route this ticket.", ["billing", "refund", "account", "feature"])
jev.score(state,  "How urgent is this?", ["low", "medium", "high"])      # -> 1.9
```

One forward pass each. Nothing is generated. The output is a typed value with a probability.

## On the name

**It is a tease and a tribute, in that order.**

`myNameisJevToo` is a joke about how many "Jev, but mine" projects appeared within a week of
launch, and it is a genuine nod to the team at **TypeSafe AI**. Jev made *"typed decisions with
calibrated probabilities"* into a category worth building in, and it did so with a clean,
inspectable contract rather than a wall of marketing: state in, `choice` / `noul` / `score` out,
output tokens free. That contract is the good idea here, and this repository is our attempt to
see how far the *interface* alone gets you on weights anyone can download.

This project is **not affiliated with, endorsed by, or connected to TypeSafe AI**. "Jev" and
"System One" are their names for their model and model class. Every Jev figure quoted below is
their own published number, cited to where they published it.

## The technique

A decoder-only LM does not need to generate text to answer a question. It applies a causal mask,
so the computation at position `t` sees only positions `< t` — which means `logits[t]` is already
a complete distribution over the next token, computed without having seen the future.

Put the state and the candidate answers in the context, end it where the answer belongs, and read:

```python
p_last = softmax(model(input_ids)[0, -1, :])   # one pass, whole state
```

Three details turn that into something usable:

1. **Make each option a single token.** Write options as `" A"`, `" B"`, `" C"`. Each is one token
   in most tokenisers, so every option's probability is read straight off `p_last`. For N options
   the total cost is **one** forward pass, no matter how large N is.
2. **Renormalise over the option set.** `share = raw / raw.sum()` answers "given that it picks one
   of these, which?" `raw` answers "does it want to answer at all?" — use the first to choose, the
   second to gate.
3. **Prefill once.** If options are not single tokens, clone the KV cache rather than re-running
   the state per option.

Full method: [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) ·
Everything that went wrong on the way: [`docs/PITFALLS.md`](docs/PITFALLS.md)

## Results

All measurements on a **Mac mini M4, 16 GB, no accelerator**, `openbmb/MiniCPM5-2B-MLX` at 4-bit,
unless stated otherwise.

### The technique itself: verified — and one of its properties is a trap

| property | measured |
|---|---|
| Single-token labels | ` A` ` B` ` C` … ` I` are each 1 token (ids 359/408/371/437/440/438/485/417/354) |
| N-way decision cost | 1 forward pass, zero extra passes for the labels |
| Cached-prefix vs re-prefill | **4.2× faster**, bit-identical (max \|Δ mean logprob\| = **0.00000000** nats) |
| MCQ accuracy, 55 items, 4 options | **89.1 %** (87.3 % plain scaffold) |
| Calibration, letter readout | **ECE 0.040**, 1/55 confidently wrong (p ≥ 0.90) — with no fitting at all |
| Decision latency, short state | **54.5 ms** per 4-way decision |
| 4-bit vs bf16 ranking | Spearman ρ 0.95, 0/25 argmax flips on factual items |
| **Two-option questions** | **the label order decides the answer** (see below) |

#### Two options is a trap, not a test

On a binary question whose state plainly contains the answer — the ticket says *"I need this
refunded today"* — the readout does this:

| labels | with state | state withheld |
|---|---|---|
| `["no", "yes"]` | no **91.5 %** | no 94.7 % |
| `["yes", "no"]` | yes **99.2 %** | yes 89.3 % |

The answer is the *first option* in both orders, and it is the first option whether or not the
state is present. There is no content signal here at all: the model is reading the label
position, not the ticket.

This also corrects a result we initially misread as good news. On JevBench's 74 public binary
items, accuracy is 54.1 % forward and 52.7 % reversed — which looked like order *robustness*
(a Δ of 1.4 points) until the flip test showed why: chance for two options is 50 %, and a model
that always picks position A scores ≈50 % in any order. **The stable number was the symptom.**

Four- and five-option items do carry real signal (77.1 % against 28.4 % chance on the easy tier),
so this is specific to small option sets, where a position prior has nowhere to spread.


### On JevBench's frozen items: not good

Run on [JevBench](https://github.com/fstandhartinger/jevbench)'s public task items with their
published scoring formulas. **Public halves only** — 48/72 easy, 72/96 standard, 111/220 hard —
so this is not directly comparable to a full board row, and the judge tier is not public at all.

| tier | n | accuracy | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 77.1 % | 28.4 % | 68.0 |
| standard | 72 | 50.0 % | 31.7 % | 26.8 |
| hard | 111 | 39.6 % | 33.6 % | **9.0** |

| axis | ours | for reference (JevBench v1.3.0's own board) |
|---|---|---|
| Intelligence | **27.4** | Jev 1.13.0 **85.7** · jeff (GLiFormer 400M) **46.9** · SemIf (Qwen3.5-4B) **79.0** |
| Calibration | **40.2** | Jev 82.7 · jeff 64.6 |
| Speed | **73.6** | Jev 83.3 · jeff 63.5 |
| Cost | local, not comparable | Jev 52.0 · jeff 76.6 |
| **Score equivalent** | **≈13** | Jev **74.4** · jeff **54.4** · SemIf **73.1** |

Latency on these items: raw p50 **188 ms**, p95 **4092 ms** (the long hard items run to 3,746
tokens of state). Under JevBench's own-server adjustment (×2 + 0.15 s) that is p50 0.53 s,
p95 8.33 s.

Hard-tier ECE is **0.2989** against a noise floor near 0.030 — roughly ten times the floor. The
model is not slightly overconfident, it is badly overconfident: in the 0.90–1.00 confidence bin
it is right **45.0 %** of the time.

### The control that decides it

The [jev-calibration-audit](https://github.com/jujumilk3/jev-calibration-audit) found that on
MMLU-ProX, shown only the options with the state replaced by a placeholder, Jev still scored
0.383 / 0.463 against a chance rate near 0.15 — *"a third to a half of apparent multiple-choice
accuracy on this benchmark is recoverable from the option list alone."*

So we ran the same control. It is the most important number in this repository:

| tier | with state | state withheld | verdict |
|---|---|---|---|
| easy | 77.1 % | 31.2 % | state drives it |
| standard | 50.0 % | 27.8 % | partly option-list driven |
| hard | 39.6 % | **45.9 %** | **mostly option-list driven** |

On hard items the model scores **higher with the state removed**. The state does not merely fail
to help — it actively hurts. The 27.4 intelligence axis is therefore largely an artifact of
option-list priors, and true state-driven performance on hard decisions is close to zero.

Without this control the headline would have been "27.4 intelligence, above chance on every tier".
With it, the honest headline is: **on hard items, this model is guessing.**

## What this is and is not

**Is:** a small, dependency-light implementation of a real interface — read the distribution, do
not sample from it — with the harness to measure whether the result is trustworthy, including the
controls most write-ups skip.

**Is not:** a Jev-class decision model. A general-purpose language model read this way does not
become one. The board already said so before we ran anything: the systems at the top of JevBench
(Jev 85.7, SemIf 79.0, djev, Winnow) were **trained** for this, and the ones that are just LMs
with a readout sit far below them. Our 2B model lands below jeff, a 400M model — because jeff was
built for the job and ours was not.

The gap is training, not interface. That is the finding, and it is worth more than a flattering
number would have been.

Three further limits, stated plainly:

- **It cannot compute.** These are judgment reads. Anything with an exact answer belongs in code,
  handed over as facts.
- **A wrong-but-valid answer is still wrong.** The typed interface removes malformed output. It
  does not remove error, and it does not remove confident error.
- **Calibration is task-local.** Every number here is ours, on our items, on our hardware. It does
  not transfer to your task because it was measured on ours.

## Install and use

```bash
pip install -e ".[mlx]"          # core package is stdlib-only; MLX backend is optional
./datasets/fetch_jevbench.sh     # pull JevBench's public items
```

```python
from jevtoo import convert, ece, abstain_curve

jev = convert("openbmb/MiniCPM5-2B-MLX")

d = jev.choice(state, "Which exclusion applies?", LABELS, criteria=RUBRIC)
print(d.choice, d.confidence, d.share)
print(d.as_dict())          # {"labels": [...], "choice": "...", "confidence": 0.91, "share": {...}}

# gate on it rather than trusting it
print(abstain_curve([(x.confidence, x.choice == gold) for x, gold in pairs]))
```

### Layout

```
jevtoo/            the package
  readout.py         Distribution, rendering, token diagnostics  (stdlib only)
  backends.py        MLX backend + the BOS and KV-cache rules that matter
  decide.py          convert(): choice / noul / score primitives
  calibrate.py       Platt fit, ECE, noise floor, abstain curve
benchmarks/        the measurement harness, and raw JSON for every number above
  letter_readout.py  single-token label readout + timing
  mcq_eval.py        55-item MCQ, position bias, calibration, abstain curve
  token_scoring.py   per-token scores, anomaly localisation
  jevbench.py        JevBench public items, their scoring formulas
  stateblind.py      the option-list leakage control
  binary_flip_test.py  does swapping two options change the answer?
  cache_equivalence.py  proof the cached path == the naive path
  quantization.py    bf16 vs 4-bit
  latency.py / length_sweep.py
tests/run_tests.py   12 unit tests, no external test dependency
docs/
  TECHNIQUE.md       the conversion method, step by step
  PITFALLS.md        nine ways to get a plausible wrong number
  RESULTS-minicpm5.md  full write-up of the MiniCPM5-2B run
```

### Reproduce

```bash
python tests/run_tests.py                  # seconds, no weights needed
python benchmarks/binary_flip_test.py      # seconds - the order-flip check
python benchmarks/mcq_eval.py              # the 55-item calibration suite
python benchmarks/stateblind.py            # ~4 min: the leakage control
python benchmarks/jevbench.py              # ~7 min: 231 items + reversed pass
```

## Credits

- **TypeSafe AI** — Jev and the System One framing. The primitives, the pricing model and the
  public eval methodology are theirs. The name of this repository is an affectionate dig at them.
- **fstandhartinger/jevbench** — the cross-model board, the frozen task set and the scoring code
  we reproduce. The failure modes we test for are ones they documented first.
- **jujumilk3/jev-calibration-audit** — the state-blind and abstain-option findings that shaped
  the controls here.
- **OpenBMB** — MiniCPM5-2B (Apache-2.0), the model this was measured on.
- **convaiinnovations/laya**, **logan-markewich/jeff**, **Fastino/GLiNER2** — the open decision
  models that show what dedicated training buys on the same board.

## License

MIT — see [LICENSE](LICENSE), including notes on third-party material. No model weights are
vendored here.
