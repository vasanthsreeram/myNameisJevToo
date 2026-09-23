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

All measurements are ours, from real runs. Baseline model
**`openbmb/MiniCPM5-2B-MLX` at 4-bit on a Mac mini M4 (16 GB, no accelerator)**; the 27B section
below was run on a **Mac Studio (M3 Ultra, 96 GB)**. Every number is reproducible from
`benchmarks/`, and per-item detail for the JevBench runs is committed under
`benchmarks/results/`.

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

## The same technique on a 27B: model quality was the bottleneck, not the interface

The 2B results above raised the obvious question — is the interface weak, or the model?
So the identical code was run against **Qwen3.8-27B** on a Mac Studio (M3 Ultra, 96 GB), in
**both** 4-bit MLX and GGUF Q4_K_M, on the same public items with the same readout.

| tier | n | MLX 4-bit | GGUF Q4_K_M | chance |
|---|---|---|---|---|
| easy | 48 | **100.0%** | **100.0%** | 28.4% |
| standard | 72 | 94.4% | **97.2%** | 31.7% |
| hard | 111 | 69.4% | **74.8%** | 33.6% |
| **intelligence** | | **77.6** | **82.6** | |

| | MiniCPM5-2B | Qwen3.8-27B MLX | Qwen3.8-27B GGUF | Jev 1.13.0 |
|---|---|---|---|---|
| Intelligence | 27.4 | 77.6 | **82.6** | 85.7 |
| Calibration | 40.2 | **86.9** | 77.2 | 82.7 |
| Speed | 73.6 | 67.1 | 65.9 | 83.3 |
| hard-tier ECE | 0.2989 | **0.0655** | 0.1139 | — |
| Score equivalent | 13.0 | **76.8** | 74.9 | 74.4 |

Two things worth noting:

- **Intelligence went 27.4 → 82.6 on an unchanged interface.** The conversion never was the
  bottleneck. If you want a decision model, pick a better base model before you write any
  glue code.
- **Calibration and intelligence trade off between the two quantisations.** MLX 4-bit is
  markedly better calibrated (86.9 vs 77.2, ECE 0.0655 vs 0.1139) while GGUF Q4_K_M is
  smarter (82.6 vs 77.6). Neither dominates, and the difference is not visible from the
  headline accuracy numbers.

### The state-blind control on the 27B

Kept % = how much accuracy survives when the state is replaced by a placeholder:

| tier | MiniCPM5-2B | Qwen3.8-27B |
|---|---|---|
| easy | 41% | **33–35%** |
| standard | 56% | **31%** |
| hard | **116%** | **71–75%** |

The 2B scored *above* chance on the hard tier with the state removed — it was reading option
priors, not input. The 27B drops to 71–75% kept there: still real option-list reliance on hard
items, but no longer a model that ignores its input.

### Caveats

Public halves only (48/72 easy, 72/96 standard, 111/220 hard); the judge tier is not public, so
tier weights are renormalised over three tiers. The cost axis is excluded, which is why the score
equivalent is over three axes and cannot be read as a JevBench Score. Per-item detail is in
`benchmarks/results/*_detail.json`.

## What this is and is not

**Is:** a small, dependency-light implementation of a real interface — read the distribution, do
not sample from it — with the harness to measure whether the result is trustworthy, including the
controls most write-ups skip.

**Is not:** a way to make a *small* model smart. The 2B experiment is unambiguous: read this
way, it scores 27.4 intelligence and on hard items does **better with the state removed** than
with it. It is guessing, and the interface cannot fix that.

**But it is not a dead end either.** The identical code on a 27B scored **82.6** intelligence —
within 3 points of Jev's published 85.7 — with **better calibration on the 4-bit MLX leg (86.9
vs 82.7)**, and 100% on the easy tier. So the honest conclusion is not "the interface is weak"
and not "scale fixes everything", but:

> The interface is real and cheap. What it delivers is bounded almost entirely by the base
> model. Below roughly 10B parameters, read this way, a general LM is an option-list guesser;
> at 27B it is a credible decision model.

That reframes the build order. You do not need a bespoke architecture or a training run to get
Jev-shaped behaviour — you need enough base model, and then the interface gets you the rest.
The systems at the top of JevBench were trained for the job; the finding here is that you can
get close without that, which is a lower bar than the board implies.

Three limits that survive the 27B result, stated plainly:

- **It cannot compute.** These are judgment reads. Anything with an exact answer belongs in code,
  handed over as facts. The JevBench hard tier rewards exactly that, and it is the tier where
  every model here scores worst.
- **A wrong-but-valid answer is still wrong.** The typed interface removes malformed output. It
  does not remove error, and it does not remove confident error.
- **Calibration is task-local.** Every number here is ours, on our items, on our hardware. It
  does not transfer to your task because it was measured on ours.

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
