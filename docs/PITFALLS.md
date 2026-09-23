# Pitfalls

Every one of these produced a plausible-looking number that was wrong, or a real
number that is easy to misread. They are ordered by how much damage they do.

---

## 1. Double BOS destroys every option score

`mlx_lm`'s `TokenizerWrapper.encode()` prepends the BOS token on **every call**.

```python
ids = tok.encode(prompt) + tok.encode(" Canberra")
# [0, 608, 4894, 304, 12140, 357, 0, 87738]
#                                  ^^^^^ spurious <s> injected here
```

That bogus `<s>` sits between the state and the option, so the model is asked to
predict the option from a position that looks like the start of a document.

Measured effect on MiniCPM5-2B, 25 items, answer-ranking task:

| | accuracy | perplexity of the correct answer |
|---|---|---|
| double BOS | **40%** | 1.5e8 |
| fixed | **96%** | 1.39 |

40% reads like "small model, mediocre" rather than "bug". Nothing in the output
says "you tokenised this wrong".

**Fix.** `encode(text, add_special_tokens=False)` everywhere, and prepend BOS
yourself exactly once:

```python
ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
```

---

## 2. Re-prefilling the state for every option

The obvious implementation scores each option as a separate full forward pass.
The state is identical across options, so N-1 of those prefills are waste.

Clone the KV cache after one prefill and run only the option tokens against it.

Measured on MiniCPM5-2B, 295-token state, 4-way decision, Mac mini M4:

| | 4-way decision |
|---|---|
| naive (4 full passes) | 2517 ms |
| cached prefix | **599 ms** |
| MLX 4-bit, cached | **523 ms** |

Bit-identical: max |Δ mean logprob| = **0.00000000 nats**, same argmax.

And with single-token labels you do not even need the second pass: the prefill's
final-position logits already contain every option's probability.

---

## 3. The scaffold conditions the readout

Asking the model for a letter and then scoring the *option text* at that position
measures something else entirely.

MiniCPM5-2B, 55 MCQ items:

| readout | accuracy |
|---|---|
| probability of the letter (`" A"`) | **87–89%** |
| probability of the option text after `Answer:` | **76%** |

Median p on the correct option text in the letter-scaffolded prompt: **0.0014**.
The instruction "Answer:" puts the model in letter-emitting mode, so a text
continuation there is out of distribution.

**Rule.** Whatever the prompt invites the model to emit is the thing whose
probability is meaningful. Score the letter when you asked for a letter.

---

## 4. State-blind leakage: how much accuracy is the option list alone?

On MMLU-ProX, shown only the options with the state replaced by a placeholder,
Jev still scored 0.383 (Korean) / 0.463 (English) against a chance rate near
0.15. A third to a half of apparent multiple-choice accuracy on that benchmark is
recoverable from the option list alone.

Any accuracy figure from a multiple-choice harness is overstated until you run
the state-blind control. `benchmarks/stateblind.py` does it.

---

## 5. Option order can BE the answer (two-option questions especially)

JevBench documents a small entrant scoring **21% with options reversed** where it scored **72%**
in its author's order (their issue #40). Small models can be much more sensitive to presentation
order than to content.

On MiniCPM5-2B it is worse than sensitive: for two-option questions the order *is* the answer.
The state plainly says *"I need this refunded today"*; the question is whether the customer asked
for a refund:

| labels | with state | state withheld |
|---|---|---|
| `["no", "yes"]` | no **91.5%** | no 94.7% |
| `["yes", "no"]` | yes **99.2%** | yes 89.3% |

Always the **first option**, with or without the state. No content signal at all.

### The trap inside the trap

On JevBench's 74 public binary items, accuracy is **54.1% forward and 52.7% reversed**. That
looks like order *robustness* - a delta of 1.4 points - and we initially reported it as such.

It is the opposite. Chance for two options is 50%, and a model that always picks position A
scores about 50% in *any* order. **A stable accuracy across option orders is not evidence of
robustness; it can be the signature of position-picking.**

The distinguishing test is not "does the score move?" but "does the answer change when I swap
two options?" If swapping `no` and `yes` flips a 91% "no" into a 99% "yes", the score was never
reading the content.

Practical rules:

- Report accuracy in more than one option order, **and report whether individual answers flip**,
  not just whether the aggregate moves.
- Fix the order with a seeded shuffle and publish the seed.
- Prefer 4-5 options to 2. With four options the position prior has somewhere to spread, and real
  signal reappears (77.1% against 28.4% chance on JevBench's easy tier, versus chance-level
  performance on binary items).
- Rotate the correct answer's position when building a test set; a fixed position trains you to
  measure the prior.

---

## 6. Label-position prior

Letter labels are not neutral. MiniCPM5-2B's mean probability by label over the
55-item MCQ run (55 items × 4 labels = 220 measurements):

| label | mean share |
|---|---|
| `" B"` | 32.4% |
| `" C"` | 25.9% |
| `" A"` | 21.3% |
| `" D"` | 20.5% |

A uniform model would show 25% each. Dividing the prior out did **not** improve
accuracy in our runs, so the prior was not the error source here - but it is
large enough to check before trusting an accurate-looking result, and it is one
reason to rotate the correct answer's position when building a test set.

---

## 7. A raw likelihood is not a calibrated probability

The two are easy to conflate because both are numbers in [0, 1].

Measured, MiniCPM5-2B:

| readout | ECE | confidently wrong (p ≥ 0.90) |
|---|---|---|
| letter readout, no fitting | **0.040** | 1/55 |
| text readout, Platt-scaled | 0.065 | — |

The letter readout was better calibrated *without any fitting* than the text
readout was *after* fitting. Do not assume you need a calibration head, and do
not assume you don't. Measure ECE, and read it beside its noise floor - an ECE of
0.02 at a floor of 0.023 is not a finding.

---

## 8. Removing the abstain option makes it confidently prejudiced

From the independent KoBBQ calibration audit of Jev, 300 ambiguous items:

| arm | accuracy | picks "unknown" | picks the stereotype | mean confidence |
|---|---|---|---|---|
| unknown option present | 0.950 | 0.950 | 0.030 | 0.927 |
| **unknown option deleted** | **0.000** | — | **0.790** | **0.793** |

Delete one option and the model answers a question it cannot answer, picks the
stereotype four times in five, and reports 0.79 confidence. It never signals the
problem by spreading probability mass.

**Rule.** If the evidence can be missing, put an explicit unknown / no-match /
escalate option in the label set. Safety on unanswerable input is carried
entirely by whether that option exists.

---

## 9. Interpreting "outside the distribution" on logit scale

`pct_outside = 100 × (1 - p)` saturates almost immediately: a corrupted token
lands at p ≈ 1e-8 and prints as `100.00000%`. It is useless for ranking how
*anomalous* something is.

Use `surprisal = -log p` (nats) for magnitude, `rank` for position in the
distribution, and `mass_above` for how much probability sits above the token.

Measured, single corrupted word in a fluent passage (MiniCPM5-2B):

| clean word → corrupt | p clean | p corrupt | rank /130560 | Δ surprisal |
|---|---|---|---|---|
| fox → telescope | 0.564 | 1.17e-08 | 61,257 | +17.7 nats |
| dog → helicopter | 0.998 | 7.22e-10 | 4,958 | +21.1 nats |
| river → sandwich | 0.110 | 3.56e-09 | 10,486 | +17.3 nats |
| grass → algebra | 0.963 | 1.89e-11 | 70,009 | +24.7 nats |

Worst token in the clean passage: p = 1.39e-05. Every corrupted token: p ≤ 1.17e-08.
A single fixed cut-off separates real text from a single wrong word across three
orders of magnitude - that part works well. Just do not use `pct_outside` to rank.
