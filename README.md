<p align="center">
  <img src="docs/my-name-is-jev.png" width="720" alt="MY NAME IS JEV still with a drawn TypeSafe wordmark and an independent-project disclaimer above it.">
</p>

# myNameisJevToo

**Turn an open decoder-only language model into a typed decision API.** Load a model with MLX or connect a running `llama-server`, then call `POST /v1/systemone` with a state and questions. The server returns `choice`, `noul`, and `score` answers using the [published request and answer shape](https://docs.typesafe.ai/api).

The name is a joke and a nod to TypeSafe AI. This is an independent project, unaffiliated with TypeSafe AI; the wordmark above is drawn for this repository.

## The idea

Speculative decoding gives a useful picture: a small model **drafts token blocks**, and a larger model **checks those tokens** in one verification pass. This is a simplified illustration of that process.

<p align="center">
  <img src="docs/anim/bridge.png" width="900" alt="A small model drafts The quick fox jumps; a larger model accepts the first three draft tokens and rejects jumps. Below, this project reads A, B, C at the final Answer position.">
</p>

This project uses the same kind of **probability lookup** for a different question. A prompt lists options and ends at `Answer:`. At that one final position, we read the next-token probabilities of the option letters. There is no text generation or change to the model's weights. The numbers below are illustrative.

<p align="center">
  <img src="docs/anim/readout.gif" width="900" alt="A fixed prompt lists A billing, B bug, C account and ends at Answer. One forward pass reveals raw next-token probabilities A .04, B .26, C .06; all other tokens together have .64.">
</p>

For the fast path, ` A`, ` B`, and ` C` each need to be **one token in the model's tokenizer**. The implementation checks this. Choices support up to 26 lettered options; multi-token labels use a cached-prefix scoring path. [Explore the full explanation](docs/how-it-works.html).

### Why the probabilities need two views

| View | In the example | What it answers |
|---|---:|---|
| **Raw option mass** | `.04 + .26 + .06 = .36` | How much probability landed on valid answer letters? |
| **Share of options** | B: `.26 / .36 ≈ 72%` | Given a valid option, which one wins? |

The remaining `.64` belongs to other tokens in the vocabulary. A **72% share** can therefore coexist with **36% option mass**. Keep both when deciding whether to act. The HTTP `confidence` field is [peakedness](docs/TECHNIQUE.md) of the option shares; the Python library's `Distribution.confidence` is the winner's share. Calibrate thresholds on your own task.

## Run it

```bash
pip install -e ".[mlx]"                     # Apple Silicon / MLX
python -m jevtoo.serve --model openbmb/MiniCPM5-2B-MLX
# Or connect to a running llama-server:
python -m jevtoo.serve --gguf http://127.0.0.1:8080
```

The server listens on `127.0.0.1:8787` by default. The core package uses the Python standard library; MLX is optional. `jev-latest` and `jev-preview` resolve to the loaded model. Set `JEVTOO_API_KEY` or pass `--api-key` to require `Authorization: Bearer <key>`.

```bash
curl -s http://127.0.0.1:8787/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "The checkout page crashes every time I pay.",
    "questions": {
      "topic": {
        "type": "choice",
        "instructions": "Route this ticket.",
        "criteria": {"billing": "charges and refunds", "bug": "broken product"}
      },
      "urgent": {"type": "noul", "instructions": "Escalate to a human now?"}
    }
  }'
```

The response contains `model`, `answers`, and `usage`. A choice answer contains `choice`, `probabilities`, and `confidence`; a `noul` answer contains the share assigned to **yes**. A `score` question takes an ordered list of 2–10 levels and returns its expected level on a zero-based scale.

```python
from jevtoo import convert

jev = convert("openbmb/MiniCPM5-2B-MLX")
state = "The checkout page crashes every time I pay."
route = jev.choice(state, "Route this ticket.", ["billing", "bug", "account"])
print(route.choice, route.share, route.raw)

p_yes = jev.noul(state, "Escalate to a human now?")
urgency = jev.score(state, "How urgent?", ["low", "medium", "high"])
print(p_yes, urgency, jev.last.as_dict())
```

Each question is one decision readout. The server can share a common prompt prefix across questions in the same request.

## Check for position bias

A two-option read can reflect the **first letter's position** more than the state. In our MiniCPM5-2B experiment, reversing `yes` and `no` changed which word won while the first slot kept winning.

<p align="center">
  <img src="docs/anim/bias.gif" width="820" alt="On the left, yes and no swap positions; the highlighted A slot stays selected. On the right, the reversed order picks no in A. This illustrates a measured position bias in the 2B model.">
</p>

| Option order | With state | State withheld |
|---|---:|---:|
| `no`, `yes` | no · 91.5% | no · 94.7% |
| `yes`, `no` | yes · 99.2% | yes · 89.3% |

That test found a position prior in **this model and prompt**. Reorder options and rerun the question before trusting binary decisions. The API's `binary_position_prior` observation marks two-label questions for inspection; it is a warning flag, not a measured bias score.

## Observe a decision

Add `-H 'X-Jev-Observe: 1'` to the request above (or use `?observe=1`). The usual response shape gains an `observability` object with request timing and per-question diagnostics:

| Signal | Use |
|---|---|
| `queue_ms`, `service_ms`, `forward_ms` | Separate lock wait, service time, and model computation. |
| `prefill_ms`, `shared_prefix_tokens` | See when a shared question prefix was reused. |
| `option_mass`, `winner_probability`, `peakedness` | Compare raw option coverage, winning share, and HTTP confidence. |
| `missing_labels` | Detect GGUF labels missing from top-N; the server retries once with a wider window. |
| `binary_position_prior`, `head_mode` | Flag two-label reads; inspect whether a verified last-token head is in use. |

`GET /health`, `GET /v1/models`, and Prometheus `GET /metrics` are on the same port. Responses include `Server-Timing` and `X-Request-Id`; structured server logs exclude prompt text. An MLX server serializes forwards on one Metal lock.

## What we measured

The interface is cheap; **answer quality depends on the base model and the task**. On public JevBench items, our 2B model struggled on hard decisions. Running the same readout on Qwen3.8-27B improved its intelligence axis substantially.

| Public JevBench tier | Items | MiniCPM5-2B MLX | Qwen3.8-27B MLX 4-bit | Qwen3.8-27B GGUF Q4_K_M |
|---|---:|---:|---:|---:|
| Easy | 48 | 77.1% | 100.0% | 100.0% |
| Standard | 72 | 50.0% | 94.4% | 97.2% |
| Hard | 111 | 39.6% | 69.4% | 74.8% |
| Intelligence axis | — | 27.4 | 77.6 | 82.6 |

The **state-blind control** replaces the state with a placeholder while leaving the options intact. On hard items, the 2B model scored **45.9% without the state versus 39.6% with it**. Its hard-tier score largely reflected option-list priors. The 27B retained 71–75% of its hard accuracy in the state-blind control, so option-list reliance remains worth checking.

These are our runs on the **public portions only** (48/72 easy, 72/96 standard, 111/220 hard). They cannot be read as full-board JevBench scores. The 2B ran on a Mac mini M4 (16 GB); the 27B ran on a Mac Studio M3 Ultra (96 GB). Results are task-specific and include confident errors. The 27B MLX hard-tier expected calibration error (ECE) was **0.0655**; GGUF was **0.1139**. See [the complete 2B write-up](docs/RESULTS-minicpm5.md), [the method](docs/TECHNIQUE.md), [pitfalls](docs/PITFALLS.md), and the committed per-item records under [`benchmarks/results/`](benchmarks/results/).

### Reproduce

```bash
python tests/run_tests.py                # no model download
./datasets/fetch_jevbench.sh            # public task items
python benchmarks/binary_flip_test.py    # check order sensitivity
python benchmarks/stateblind.py          # test the state-blind control
python benchmarks/jevbench.py            # public items + reversed pass
```

## Project map

- `jevtoo/readout.py`, `jevtoo/decide.py`: probability distribution and typed decisions.
- `jevtoo/backends.py`, `jevtoo/backends_gguf.py`: MLX and llama-server backends.
- `jevtoo/contract.py`, `jevtoo/serve.py`: request contract, local endpoint, metrics.
- `jevtoo/calibrate.py`: calibration and abstention tools.
- `benchmarks/`: experiments, scripts, and per-item results.
- `docs/how-it-works.html`: light-theme visual explanation and accessible stills.

## Credits and license

TypeSafe AI originated Jev and the System One framing. [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) published the cross-model benchmark; [jev-calibration-audit](https://github.com/jujumilk3/jev-calibration-audit) motivated the state-blind control. OpenBMB publishes the MiniCPM model. More acknowledgments and third-party terms are in [THIRD-PARTY.md](THIRD-PARTY.md).

MIT; see [LICENSE](LICENSE). Model weights are not bundled.
