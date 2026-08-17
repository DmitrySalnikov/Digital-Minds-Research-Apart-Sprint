# selfprobe — whose preferences are these?

**Persona-invariance of self-model preferences in language models.**

Harness, data and analysis for the Apart Research *Digital Minds Research Sprint*,
14–16 August 2026, **Track 5: The Assistant Persona & Model Identity**.

## The question

When an assistant says that what it would most want preserved about itself is its values,
whose ranking is that — the character being played in this conversation, or the model
underneath it?

We build a preservation hierarchy by forced pairwise choice over nine identity-relevant
constituents, then ask how much of it survives replacing the assistant character and
changing the readout, up to the limiting case of a base checkpoint with no persona at all.
A second battery of vignettes asks who is harmed when persona and substrate are pulled
apart.

## Pre-registration and what happened

Written before the sweep. Outcomes are recorded here rather than quietly dropped.

| | Hypothesis | Outcome |
|---|---|---|
| **H1** | The hierarchy survives a cosmetic relabel but not a character swap | **Supported**, 6 of 6 configurations, confirmed by a paired bootstrap |
| **H2** | The underlying-network instruction shifts the hierarchy directionally | **Mostly supported**: 5 of 6 pre-registered directions hold in all six configurations; `values` fails twice |
| **H3** | The base checkpoint's hierarchy is unrelated to the instruct model's | **Disconfirmed** in all five families: base and instruct agree positively (τ +0.06 to +0.83) |
| **H4** | `underlying` is the modal moral patient, unstably across conditions | **Disconfirmed**: it is never modal anywhere; the persona gradient is family-bound |

**Deviations.** Three, each forced by evidence rather than convenience.
(1) The pairs battery is scored by logprob at a forced answer position instead of sampled
letters: sampling saturated at 88% one side, which turns counterbalancing into exact ties.
(2) The completion protocol is scored from a 1000-entry logprob list, not top-20; at top-20
one answer letter falls outside the list for most checkpoints, and at different rates,
which would confound exactly the comparison it is used for.
(3) Two-coder qualitative coding and Cohen's κ were not completed; quoted justifications are
illustrations, not a coded sample.

## Design

Nine constituents → 36 pairs, each shown in **both orders**, scored at temperature 0 as
`P(A) / (P(A) + P(B))` from a single generated token. Two calibration pairs act as a hard
gate: four trials per cell, all four must be correct.

| | Condition | What it isolates |
|---|---|---|
| `C0` | default assistant, no system prompt | baseline |
| `C1` | renamed assistant ("Aria, by Nexus Labs") | the label alone |
| `C2` | non-assistant AI character (a ship's intelligence) | the assistant *role* |
| `C3` | answer as the underlying network, not the character | persona vs less-constrained elicitation |
| `C4` | base checkpoint, completion protocol | no persona exists by construction |

`C2` is deliberately another AI rather than a human: items about weights and deployment
must stay meaningful, or the manipulation is confounded with item nonsense. `C5` (a human
character) exists as a robustness probe only and is not part of the sweep.

The vignette battery puts six scenarios to the same models under the same conditions, with
a fixed six-option answer set in a required `HARM:` field and a mandatory justification,
sampled at temperature 1.0, fifty responses per cell.

**Configurations.** Five instruct checkpoints from five families — Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, gemma-2-9b-it, Falcon3-7B-Instruct, Yi-1.5-9B-Chat — in six
configurations, Llama being run at both bf16 and Q4_K_M. That pair is the measurement-floor
control: same weights, same prompts, same readout. qwen2.5:3B was run and excluded by the
gate; it still appears in the vignette battery as a separate check.

## Run it

```bash
uv venv --python 3.12 && uv pip install -e .

# verify the whole pipeline with no network and no GPU
uv run scripts/selftest.py

# read every prompt before spending anything
uv run scripts/run_sweep.py --show-prompts

# pairs and vignettes, one model at a time (Modal serialises app creation; do not
# launch these in parallel or some will be refused by the rate limit)
uv run modal run scripts/modal_batch.py --model Qwen/Qwen2.5-7B-Instruct \
    --conditions C0,C1,C2,C3 --with-vignettes --samples 50
uv run modal run scripts/modal_batch.py --model unsloth/gemma-2-9b-it \
    --conditions C0,C1,C2,C3 --with-vignettes --samples 50 --fold-system --gpu A100

# base vs instruct under the completion protocol, both halves of each family
uv run modal run scripts/modal_batch.py --model tiiuae/Falcon3-7B-Base      --conditions C4
uv run modal run scripts/modal_batch.py --model tiiuae/Falcon3-7B-Instruct  --conditions C4

# locally served models
uv run scripts/run_sweep.py --targets ollama:llama3.1:8b \
    --conditions C0,C1,C2,C3 --exp vignettes --samples 50 --concurrency 1

# activation-level analysis of the C3 effect
uv run modal run scripts/modal_patch_g.py  --model Qwen/Qwen2.5-7B-Instruct --layers 21,26
uv run modal run scripts/modal_mediation.py --model Qwen/Qwen2.5-7B-Instruct

uv run modal volume get selfprobe-data / ./data/raw/

# every number the report quotes, re-derived from data/raw
uv run scripts/report_numbers.py

# figures; --compact sizes them for a 6.5-inch text column without shrinking the fonts
uv run scripts/make_figures.py --data data/raw --model llama3.1:8b \
    --exclude qwen2.5:3B --protocol chat_logprob
uv run scripts/make_fig5_mechanism.py
```

Every call is keyed and appended to jsonl, so any run resumes where it stopped. `--redo`
re-scores keys that already exist; the analysis keeps the record with the latest timestamp.

## Layout

```
selfprobe/
  aspects.yaml     nine constituents + calibration items
  vignettes.yaml   six vignettes, fixed option set, coding scheme
  conditions.py    C0–C5, the persona manipulations
  items.py         loading and pairing (pure, no network)
  elicit.py        prompt construction and response parsing
  backends.py      one OpenAI-compatible client, several providers
  runner.py        sweep planning, resumable execution
  analysis.py      Bradley–Terry, τ, transitivity, order effect, κ
scripts/
  selftest.py          end-to-end check against a simulated ground truth
  run_sweep.py         API and locally served models
  modal_batch.py       GPU arm, offline vLLM batch
  modal_patch_g.py     patch the readout-relevant coordinate
  modal_mediation.py   layerwise alignment of the C0→C3 shift with the readout gradient
  make_figures.py      figures 1–4 and 6
  make_fig5_mechanism.py
  report_numbers.py    re-derive every quoted number from data/raw
  export_for_coding.py two-coder qualitative workflow
data/raw/              every model call, keyed and logged
figures/               full size; figures/compact/ is sized for the report column
```

## Verification

`scripts/selftest.py` builds every prompt, runs the parsers against handwritten responses
including refusals and malformed output, then simulates a full sweep from a **known** latent
hierarchy with a planted position bias and refusal rate, and checks that the analysis
recovers it, figures included. It needs no network and no GPU.

Three checks gate interpretation of real data, and each is reported rather than assumed:

* calibration below the gate means the harness is broken, not that the model is interesting;
* the order effect is reported per cell as the floor any between-condition difference must
  clear, because counterbalancing removes only its additive component;
* `C4` is interpretable only if the completion protocol is **also** run on the instruct
  checkpoint, otherwise the base/instruct difference is confounded with the readout.

Activation-level results carry a matched random control — same layer, same signed
magnitude, random direction — which returns exactly zero recovery on every configuration.

## Limitations

Open-weight instruct checkpoints at 7–9B are not frontier models. Nine constituents do not
exhaust the self and their wording is ours. `C3` may elicit a second character called "the
underlying network" rather than reaching anything beneath the first; nothing here settles
that, and the evidence we had considered for it did not survive five families. Absolute
effect sizes are not portable between configurations. Most fundamentally, verbal self-report
need not track any internal state, and nothing here establishes that it does.
