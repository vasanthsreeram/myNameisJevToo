# The conversion technique

Turning any causal language model into a System-One-style decision model.

## The claim

A decoder-only LM does not need to generate text to answer a question. It only
needs one forward pass and someone willing to read the distribution instead of
sampling from it. This document is the complete method.

No retraining. No fine-tuning. No new weights. The model is not modified at all.
What changes is the interface.

## Why it works

A causal LM applies a causal mask, so the computation at position `t` attends
only to positions `< t`. Therefore `logits[t]` is a complete distribution over
the vocabulary for the token that comes next, computed without ever having seen
the future.

Put the question and the candidate answers in the context, end the context at the
point where the answer belongs, and the final position's distribution *is* the
model's answer distribution. Nothing needs to be generated to obtain it.

```python
logits = model(input_ids)              # one pass, whole state
p_last = softmax(logits[0, -1, :])     # distribution over 130,560 tokens
```

## Step 1 - make every option a single token

This is the step that makes the read free.

Write options as labels: `" A"`, `" B"`, `" C"`. In MiniCPM5-2B each is exactly
one token (ids 359 / 408 / 371), so each option's probability can be read
directly from `p_last` with no additional forward pass:

```python
p_a = p_last[359]
p_b = p_last[408]
```

For N options, the cost is **one** forward pass total, regardless of N. The
"spec-decode read" is free precisely because the label is one token.

Check this property for your tokeniser before assuming it holds:

```python
assert len(tok.encode(" A", add_special_tokens=False)) == 1
```

If a label spans multiple tokens, fall back to the cached-prefix path in step 3.

## Step 2 - frame the state

```
<state>

<instructions>

A. <option text> - <rubric description>
B. <option text> - <rubric description>
C. <option text> - <rubric description>

Answer:
```

Include the rubric text. The model is choosing between *described* options, not
bare identifiers, and the descriptions are where the actual criterion lives.

The literal string the prompt ends with matters. Whatever the scaffold invites
the model to emit is the thing whose probability is meaningful - see
`PITFALLS.md` §3 for the measured cost of reading the wrong thing.

## Step 3 - one prefill, cached options

If options are not single tokens, do not re-run the state per option. Prefill
once, clone the KV cache, and run only the option tokens against it.

```python
cache = make_prompt_cache(model)
pre = model(mx.array([p_ids]), cache=cache)      # the state, once

for option in options:
    cache_i = copy.deepcopy(cache)               # shared prefix
    out = model(mx.array([option_ids]), cache=cache_i)
```

Measured: 4.2× faster than the naive path and bit-identical (max |Δ mean logprob|
= 0.00000000 nats).

## Step 4 - renormalise over the option set

The raw probabilities do not sum to 1, because the model spreads mass over
130,000 other tokens. Two numbers matter, and they answer different questions:

```python
raw   = p_last[label_ids]              # "does the model want to answer at all?"
share = raw / raw.sum()                # "given that it answers, which option?"
```

Use **share** to pick. Use **raw** to gate: if every option is unlikely, the
question may be out of scope, or the evidence may be missing.

## Step 5 - the three typed primitives

| primitive | question shape | what you read |
|---|---|---|
| `choice` | one of a known set | `argmax(share)`, plus the full distribution |
| `noul` | yes / no | `share[yes]` - a gate probability |
| `score` | a level on a scale | `Σ level_i × share_i` - an expected value |

```python
jev = convert("openbmb/MiniCPM5-2B-MLX")

jev.choice(state, "Route this ticket.", ["billing", "refund", "account", "feature"])
jev.noul(state,  "Has the customer been charged twice this month?")
jev.score(state, "How urgent is this?", ["low", "medium", "high"])
```

All three are one forward pass each. None generates text.

## Step 6 - calibrate, then gate

A raw likelihood is a sufficient statistic, not a probability. Whether you need
to fit anything is an empirical question - measure it:

```python
ece_value, bins = ece([(d.confidence, d.choice == gold) for d, gold in pairs])
```

Then find the operating point:

| threshold | coverage | accuracy on acted |
|---|---|---|
| 0.00 | 100.0% | 89.1% |
| 0.70 | 85.5% | 91.5% |
| 0.95 | 56.4% | 96.8% |

(MiniCPM5-2B, 55-item MCQ task.) Acting only above 0.95 trades half the traffic
for an error rate cut by 6×. That trade is application code's decision, not the
model's - which is the point of a typed decision model.

## Step 7 - verify the readout, not just the score

Three controls, in order of how much they have changed conclusions:

1. **State-blind.** Replace the state with a placeholder and keep the options. If
   accuracy survives, you measured the option list, not judgment.
2. **Reversed option order.** Small models have moved 51 accuracy points on
   option reordering (JevBench issue #40: 72% → 21%).
3. **ECE beside its noise floor.** At n=55 a floor near 0.04 makes a 0.04 ECE
   unremarkable.

## What conversion does not buy you

The honest limits of the technique:

- **It cannot compute.** These are judgment reads. Anything with an exact answer
  should be computed in code and handed over as facts.
- **A wrong-but-valid answer is still wrong.** The typed interface removes
  malformed output. It does not remove error.
- **Calibration is task-local, not a property of the model.** Numbers here were
  measured on our items, on our hardware, against our scaffolds.
- **This is not an architecture.** There is no new module, no trained head, no
  novel layer. It is a reading of what a causal LM already computes. Anyone
  describing it as a new model class is overclaiming, and so would we be.
