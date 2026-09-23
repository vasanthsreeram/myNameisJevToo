# MiniCPM5-2B as a Jev-style scorer — real measurements

Host: Mac mini M4, 16 GB, no GPU server. All numbers from actual forward passes.

## 1. What was pulled

| Repo | Format | Size | Path |
|---|---|---|---|
| `openbmb/MiniCPM5-2B` | bf16 | 4.7 G | `~/models/MiniCPM5-2B` |
| `openbmb/MiniCPM5-2B-MLX` | 4-bit affine, group 64 | 1.3 G | `~/models/MiniCPM5-2B-MLX` |

Architecture: **standard `LlamaForCausalLM`** — 42 layers, hidden 2048, 16 heads / 2 KV heads (GQA),
vocab 130,560, 131,072 context, apache-2.0, no custom kernels, no `trust_remote_code`.

That last fact is the whole reason this works: nothing is hidden behind a custom forward pass,
so the logits are directly readable.

## 2. The operation — yes, it is surgical

Not autoregressive generation. One forward pass over `[state + given answer]`; the causal mask
means position *t* only sees tokens before it, so `logits[t-1]` **is** the model's distribution
for token *t*. You never sample. You read.

Per given token `v`:

| quantity | meaning |
|---|---|
| `p` | probability the model assigns to the given token |
| `rank` | how many of the 130,560 vocab entries score above it |
| `pct_outside` | `100 × (1 − p)` — share of probability mass that disagrees |
| `mass_above` | probability mass strictly above the token |
| `surprisal` | `−log p` (nats) |
| `margin` | distance below the model's own top pick |

Cost is `O(1)` passes per answer, not `O(answer length)`.

## 3. Accuracy

### Answer ranking — correct vs 3 distractors, 25 items (`mcpm_verify.py`, Test A)

**24/25 = 96.0 % top-1 accuracy**, using length-normalised log-probability.

The single miss is genuine and interesting: *"The human heart has"* → picked
*" three chambers"* over *" four chambers"* (margin −0.93).

Median margin over the best distractor: **+2.3 nats**.

### Token-level anomaly localisation (Test B)

One word corrupted in an otherwise fluent passage — 4/4 caught:

| clean → corrupt | p clean | p corrupt | rank /130560 | pct_outside |
|---|---|---|---|---|
| fox → telescope | 0.564 | 1.17e-08 | 61,257 | 100.00000 % |
| dog → helicopter | 0.998 | 7.22e-10 | 4,958 | 100.00000 % |
| river → sandwich | 0.110 | 3.56e-09 | 10,486 | 100.00000 % |
| grass → algebra | 0.963 | 1.89e-11 | 70,009 | 100.00000 % |

Worst token in the *clean* passage: `p = 1.39e-05`.
Every corrupted token: `p ≤ 1.17e-08`.

**A single fixed cut-off separates them** — cleanest legitimate token is 3 orders of magnitude
above the most likely corrupted token. This is the part of your idea that works best.

### Calibration — can the score be a real probability? (Test D)

100 candidate points (25 correct / 75 wrong), 5-fold CV grouped by item so no leakage:

| metric | value |
|---|---|
| answer-level AUC (raw geometric p) | **0.9392** |
| best achievable accuracy (cut-off sweep, in-sample) | 93.0 % |
| accuracy @ p ≥ 0.5 after Platt scaling (out-of-fold) | 89.0 % |
| Brier, raw p | 0.0864 |
| Brier, Platt-scaled out-of-fold | 0.0752 |
| **ECE, Platt-scaled (10 bins)** | **0.0645** |

Token-level discrimination (Test C) is much weaker: **AUC 0.744**
(median p 0.784 on correct-side tokens vs 0.020 on wrong-side). So: use the score
per-token for *anomaly detection*, and per-answer for *discrimination*.

## 4. Latency (1 prefill + 4-candidate decision)

| state tokens | bf16 | 4-bit | vs Jev's advertised 70–500 ms |
|---|---|---|---|
| 51 | 282 ms | **161 ms** | inside |
| 151 | 441 ms | **297 ms** | inside |
| 301 | 603 ms | **525 ms** | borderline |
| 601 | 1011 ms | 982 ms | slower |
| 1201 | 1794 ms | 1962 ms | slower |
| 2354 | 3450 ms | 3946 ms | slower |

Jev advertises 70–500 ms. On a *local Mac mini* with no accelerators we are inside that band
up to ~300-token states.

**Critical optimisation:** naively re-scoring each candidate re-prefills the whole state.
Prefilling once and scoring candidates against the cloned KV cache is **4.2× faster and
bit-identical** — max |Δ mean log-prob| = 0.00000000 nats, same argmax.
2.52 s → 0.60 s (bf16), 2.75 s → 0.52 s (4-bit).

## 5. Quantization

| metric | value |
|---|---|
| Spearman ρ of mean log-prob, bf16 vs 4-bit | 0.9519 |
| mean \|Δ mean log-prob\| | 0.685 nats |
| max \|Δ\| | 2.914 nats |
| argmax answer changed (factual items) | 0/25 |
| argmax answer changed (ambiguous routing state) | **yes — flipped** |

Verdict: 4-bit is safe for *ranking*, not for *absolute calibrated probabilities*.
Use bf16 if you want the number to mean something; 4-bit if you want a fast route.

## 6. The bug that would have fooled anyone

First run reported **40 % accuracy** with perplexities around 10⁸ — plausible-looking garbage.

Cause: `TokenizerWrapper.encode()` prepends BOS on **every** call. Concatenating
`encode(prompt) + encode(answer)` injects a bogus `<s>` between them, which obliterates the
score of the first answer token.

Fix: `encode(text, add_special_tokens=False)` and add BOS exactly once at the front.
That alone took accuracy from 40 % → 96 %.

Lesson: a 40 % headline reads like "small model, mediocre". It was a tokenisation artefact.

## 7. Honest caveats

- 25 items is a **small** evaluation set. AUC 0.939 is real but the confidence interval is wide.
- ECE 0.0645 is *not* Jev-grade calibration. Jev's pitch is calibrated typed output; getting
  there needs a few hundred labelled decisions in *your* domain plus a fitted head.
- The token-level score degrades on ambiguous text — it localises *wrong* tokens, not *wrong ideas*.
- Base model only. No fine-tuning. This is the raw checkpoint's behaviour.

## 8. Scripts

| file | purpose |
|---|---|
| `~/scripts/mcpm_score.py` | surgical per-token scorer + cached multi-candidate path |
| `~/scripts/mcpm_verify.py` | Tests A–D, writes `mcpm_verify_report.json` |
| `~/scripts/mcpm_quant_compare.py` | bf16 vs 4-bit comparison |
| `~/scripts/mcpm_latency.py` | decision latency |
| `~/scripts/mcpm_length_sweep.py` | latency vs state length |
| `~/scripts/mcpm_cached_check.py` | cached-vs-naive equivalence proof |

Run with `/opt/homebrew/bin/python3` (mlx-lm 0.31.3, mlx 0.29.3).

---

# 9. Choice readout — spec-decode style (A/B/C/D), the specified design

Model: `openbmb/MiniCPM5-2B-MLX`, **4-bit affine**. 55 items (25 hand-written + 30 generated),
correct-option position shuffled deterministically by seed.

**Mechanism.** One forward pass over the state, which already contains the choices:

```
What is the capital of France?

A. Paris
B. Oslo
C. Lima
D. Cairo

Answer:
```

Labels ` A` ` B` ` C` ` D` are **single tokens** (ids 359 / 408 / 371 / 437), so each
probability is read straight off `softmax(logits[-1])`. **Zero extra forward passes** — the
"spec decode" read is free because the label is one token. Nothing is generated; the output is
four probabilities.

## Accuracy

| method | accuracy | median score |
|---|---|---|
| L1 letter readout, plain scaffold | 48/55 = **87.3 %** | 0.921 |
| L2 letter readout + "answer with a single letter" | 49/55 = **89.1 %** | 0.960 |
| L1d prior-debiased (leave-one-out) | 48/55 = 87.3 % | — |
| L2d prior-debiased | 48/55 = 87.3 % | — |
| T text readout after `Answer:` | 42/55 = 76.4 % | −6.430 |
| H hybrid letter + text | 48/55 = 87.3 % | — |

## Position bias

| method | A | B | C | D |
|---|---|---|---|---|
| L1 | 12/13 | 11/12 | 13/13 | **12/17** |
| L2 | 11/13 | 12/12 | 13/13 | 13/17 |

Mild. Position D is the weak spot under L1 (71 %).

Letter prior, L1: `A` 21.3 % · `B` **32.4 %** · `C` 25.9 % · `D` 20.5 %. A bias toward B exists,
but dividing it out did **not** improve accuracy, so it is not the error source.

## Calibration — when it says 90 %, is it right 90 % of the time?

| method | bin | n | mean conf | accuracy | gap |
|---|---|---|---|---|---|
| L1 | [0.50,0.70) | 7 | 0.615 | 71.4 % | −0.100 |
| L1 | [0.70,0.90) | 13 | 0.821 | 76.9 % | +0.052 |
| L1 | [0.90,1.01) | 34 | 0.958 | **97.1 %** | −0.013 |
| L2 | [0.90,1.01) | 41 | 0.968 | 95.1 % | +0.016 |

**L1 ECE = 0.0400**, confidently wrong (p ≥ 0.90) **1/55**.
L2 ECE = 0.0640, confidently wrong 2/55.

That beats the Platt-calibrated text method (ECE 0.0645) *without any fitting at all*.
The letter readout is both faster and better calibrated than the text readout.

## Abstain curve (the `gate` primitive)

| threshold | coverage | accuracy on acted | wrong |
|---|---|---|---|
| 0.00 | 100.0 % | 89.1 % | 6 |
| 0.70 | 85.5 % | 91.5 % | 4 |
| 0.95 | **56.4 %** | **96.8 %** | **1** |

Act only when p ≥ 0.95 and it is right 96.8 % of the time, on 56 % of traffic.

## Latency

| readout | median per 4-choice decision | passes |
|---|---|---|
| letter | **54.5 ms** | 1 prefill, 0 extra |
| text | 117.4 ms | 1 prefill + 4 cached |

54.5 ms is **faster than Jev's advertised 70–500 ms floor**, on a Mac mini with no accelerator.

## The trap to avoid

Scoring the option **text** after `Answer:` drops to 76.4 %. The scaffold puts the model in
letter-emitting mode, so a text continuation there is out of distribution (median p on the
correct option is 0.0014). Read the **letter**, not the text, once you have framed the prompt
as multiple choice. The earlier 96 % figure came from an un-framed prompt with no option list —
a different task, not a better version of this one.

## Scripts added

| file | purpose |
|---|---|
| `~/scripts/mcpm_choice.py` | the letter readout: `choice_probs()`, `render_mcq()`, `decide()` |
| `~/scripts/mcpm_mcq_eval.py` | 55-item MCQ eval, position bias, calibration, abstain curve |
