# takumi

## Packing

Pack sequences of known lengths into bins of at most `max_length` tokens.

### Concrete packing

`first_fit` / `worst_fit` assign every sequence to a bin up front and return
`list[list[int]]` — each bin is a list of indices into `lengths`.

```python
from takumi.packing import pack

lengths = [7, 6, 5, 4, 3, 2]

bins = pack(lengths, max_length=10, algorithm="first_fit")
# [[0, 4], [1, 3], [2, 5]]

bins = pack(lengths, max_length=10, algorithm="worst_fit", tolerance=2, seed=42)
```

- `tolerance` — bucket width for the longest-first sort; lengths in the same
  bucket count as ties.
- `seed` — shuffles ties reproducibly; `None` keeps a deterministic stable sort.

### Histogram packing (reusable blueprint)

`histogram` / `histogram_batched` return a `PackingPlan`: a `{layout: count}`
blueprint plus per-length index pools. Each epoch you draw different concrete
bins from the same blueprint.

```python
from takumi.packing import pack

plan = pack(lengths, max_length=10, algorithm="histogram")

for bin_indices in plan.iter_bins(seed=0):
    ...  # [0, 5], [2, 3], [1, 4]

# different seed -> same layouts, different index placement
for bin_indices in plan.iter_bins(seed=1, interleave=True):
    ...
```

`interleave=True` shuffles individual bins together so consecutive bins vary
in shape; the default emits all bins of one layout back-to-back.

### Packing quality

```python
from takumi.packing import PackingReport

print(plan.report())                                  # from a PackingPlan
# PackingReport: 3 bins, 6 sequences (2.00 avg/bin)
#   tokens     27/30 used (efficiency 90.00%, padding 10.00%)
#   fullness   p50 90.0% | p90 90.0% | p99 90.0%
#   max_length 10

print(PackingReport.from_bins(bins, lengths, 10))     # from concrete bins
```

### Low-level functions

The raw packers behind `pack` are importable directly:

```python
from takumi.packing import (
    pack_first_fit, pack_worst_fit,
    group_by_size, pack_histogram, pack_histogram_batched,
    stream_packed_bins,
)

counts, index_pools = group_by_size(lengths, max_length=10)
templates = pack_histogram(counts, max_length=10)
# {(5, 4): 1, (6, 3): 1, (7, 2): 1}

for bin_indices in stream_packed_bins(templates, index_pools, seed=0):
    ...
```

## Masking

MLM-style corruption of token batches. Unlike packing, this half of the
package needs a tensor framework: install `takumi[torch]`, or `jax`. The
backend is picked once at import time via `TAKUMI_MASKING_BACKEND`, so set it
before importing:

```python
import os
os.environ["TAKUMI_MASKING_BACKEND"] = "torch"  # or "jax"
```

### Presets

```python
import torch
from takumi.masking import FixedMasker

masker = FixedMasker(
    mlm_prob=0.15,
    masking_prob=0.8,
    mutation_prob=0.1,
    mask_token_id=3,
    valid_token_ids=list(range(4, 100)),
    special_token_ids=[0, 1, 2, 3],
)

x = torch.randint(4, 100, (8, 200))
result = masker(x, seed=10)

result.sequences  # corrupted tokens, same shape/type as x
result.labels     # original token at selected positions, -100 elsewhere
result.masked     # bool: replaced with the mask token
result.mutated    # bool: replaced with a random valid token
result.kept       # bool: selected but left unchanged
```

Other presets, same call signature:

```python
from takumi.masking import (
    BetaMasker,        # 80/10/10 with a Beta-sampled per-call rate
    CategoricalMasker, # per-call rate from an external sampler
    DiffusionMasker,   # mask-only, rate driven by a timestep schedule
    HybridSpanMasker,  # 80/10/10 over contiguous spans plus scatter
    MaskGITMasker,     # MaskGIT training: per-row timestep, exact count
)

diffusion = DiffusionMasker(
    total_timesteps=1000,
    mask_token_id=3,
    special_token_ids=[0, 1, 2, 3],
    schedule="cosine",  # MaskGIT's; or "linear", "sqrt", or a callable
    time="discrete",    # whole-number timesteps; "continuous" for real ones
)
result = diffusion(x, seed=10, timestep=500)
result = diffusion(x, seed=10, timestep=torch.randint(0, 1001, (8,)))  # per row

maskgit = MaskGITMasker(mask_token_id=3, special_token_ids=[0, 1, 2, 3])
result = maskgit(x, seed=10)  # each row draws its own timestep
```

### ELBO loss weights (masked diffusion)

Every result also carries one value per row: `result.timesteps` (each row's
noise level, for time-conditioned models; `None` for rates without time) and
`result.weights`, the masked-diffusion ELBO weight. Multiplying each row's
summed masked-token cross-entropy by its weight gives the MDLM / MD4
objective, an upper bound on the negative log-likelihood. `ScheduleRate` (so
`DiffusionMasker`) takes a required `time`: `"discrete"` for whole-number
steps drawn from 1..T, which gets the exact discrete-time weight, or
`"continuous"` for real-valued steps, which gets `p′(t)/p(t)`.
`UniformTimestepRate` draws real `t`, so it always uses `p′(t)/p(t)`. Rates without time give weights of 1, i.e. plain MLM loss.

```python
import torch.nn.functional as F

steps = torch.randint(1, 1001, (8,))  # 1..T: step 0 masks nothing
result = diffusion(x, seed=10, timestep=steps)
logits = model(result.sequences)                       # (B, L, V)
ce = F.cross_entropy(logits.transpose(1, 2), result.labels,
                     ignore_index=-100, reduction="none")  # (B, L)
loss = (result.weights * ce.sum(1)).sum() / x.numel()  # NELBO per token
```

The weights are exact for independent per-token masking (`DiffusionMasker`
without `focus`); with exact-count masking (`MaskGITMasker`) they are an
approximation.

With real-valued `t`, rows near `t = 0` mask almost nothing and get huge
weights, which can spike small-batch training. Every schedule takes an
optional floor on the masking probability, e.g. `CosineSchedule(eps=1e-3)`:
masking and both weights then use `eps + (1 - eps) * p(t)`, so weights stay
near `1 / eps` or below. Integer timesteps are already capped at `T`.

### MaskGIT decoding

`MaskGITDecoder` runs one step of MaskGIT's iterative decoding: it commits the
model's samples, then re-masks the least confident ones so the masked fraction
follows the schedule. `backend.sample` draws one token per position from the
`(B, L, V)` logits on the model's device, so only the `(B, L)` tokens and their
probabilities reach the decoder.

```python
from takumi.masking import MaskGITDecoder, backend

decoder = MaskGITDecoder(mask_token_id=3, total_steps=12, seed=0)

tokens = torch.full((8, 200), 3)  # start fully masked
for step in range(12):
    sampled, probs = backend.sample(model(tokens), mask_token_id=3,
                                    rng=decoder.rng)
    tokens = decoder(tokens, sampled, probs, step, num_unknown=200)
```

`backend.sample` never draws `mask_token_id` and takes `temperature=0.0` for
greedy decoding. The decoder owns the run's random stream: each step draws
fresh noise from it, and `rng=decoder.rng` puts token sampling on the same
stream, so `seed` reproduces the whole run. Call `decoder.reset(seed)` before
each new run to pick its seed; `reset()` with no seed draws a fresh one, and
`decoder.seed` records it so `decoder.reset(decoder.seed)` replays the run.

#### Remasking committed tokens

Plain MaskGIT never revisits a token once it is committed, so early mistakes
stay. With `remask_prob > 0` (off by default), each token committed on an
earlier step goes back to the mask with that probability on every step whose
progress falls inside `remask_window`, and gets re-predicted later
([ReMDM](https://arxiv.org/abs/2503.00307)). Pass `current_probs` to re-mask
the tokens the model now trusts least instead of random ones. Re-predicting
costs steps, so raise `total_steps` when you turn it on.

```python
decoder = MaskGITDecoder(
    mask_token_id=3,
    total_steps=24,
    remask_prob=0.1,
    remask_window=(0.25, 0.75),  # progress range where remasking applies
    seed=0,
)

tokens = torch.full((8, 200), 3)
generated = tokens == 3  # what may be re-masked; prompt tokens are left alone
for step in range(24):
    logits = model(tokens)
    sampled, probs = backend.sample(logits, mask_token_id=3, rng=decoder.rng)
    current = backend.token_probs(logits, tokens)
    tokens = decoder(tokens, sampled, probs, step, num_unknown=200,
                     generated=generated, current_probs=current)
```

### Composing a corruptor directly

Every preset is one composition of three swappable strategies: a rate policy
(how much to select), a selector (which positions), and a corruption scheme
(what to do with them).

```python
from takumi.masking import Corruptor, BetaRate, Bernoulli, Mixed

corruptor = Corruptor(
    rate=BetaRate(mean=0.15, concentration=4.0),
    select=Bernoulli(),
    corrupt=Mixed(
        mask_prob=0.8,
        mutate_prob=0.1,
        mask_token_id=3,
        valid_token_ids=list(range(4, 100)),
    ),
    special_token_ids=[0, 1, 2, 3],
)

result = corruptor(x, seed=10)
```

Available pieces:

- rates — `FixedRate(rate)`, `BetaRate(mean, concentration)`,
  `SampledRate(sampler)`, `ScheduleRate(total_timesteps, schedule, time=...)`,
  `UniformTimestepRate(schedule)` (one rate per row)
- selectors — `Bernoulli()`, `FocusBiased(strategy, strength)`, `Spans(...)`,
  `ExactCount()`
- schemes — `Mixed(mask_prob, mutate_prob, mask_token_id, valid_token_ids)`,
  `MaskOnly(mask_token_id)`

Per-call options:

```python
result = corruptor(
    x,
    seed=10,                  # bit-exact reproducible call
    excluded_ids=[0, 1],      # overrides special_token_ids for this call
    focus=focus_mask,         # (B, L) bool mask to bias selection toward
    timestep=500,             # required by ScheduleRate (scalar or (B,)), ignored otherwise
)
```

## Resuming a run

A resumed run reproduces the original when three things are restored:

- **Curriculum.** Schedulers and `MLMProbabilitySampler` have `state_dict()` /
  `load_state_dict()`; save the dict in your checkpoint (or hand the object to
  Accelerate's `register_for_checkpointing`). Schedulers refuse plain pickling
  because their step is shared across DataLoader workers; loading the state
  updates every worker, running or not.
- **Data order.** An epoch is a pure function of its seed, so seed each epoch
  (e.g. `seed=epoch`) and skip the bins already consumed:
  `itertools.islice(plan.iter_bins(seed=epoch), bins_done, None)`.
- **Masking.** Derive each call's seed instead of drawing it, e.g.
  `seed_from_components(base_seed, epoch, index)`, so the same sample gets the
  same masks after a restart.

```python
# save
checkpoint["curriculum"] = sampler.state_dict()

# resume
sampler.load_state_dict(checkpoint["curriculum"])
```

## The C++ part

The performance-critical packers (`pack_first_fit`, `pack_worst_fit`,
`group_by_size`, `pack_histogram`, `pack_histogram_batched`) are written in C++
([src/packing/packing.cpp](src/packing/packing.cpp)) and exposed to Python with
[nanobind](https://nanobind.readthedocs.io/) as the compiled module
`takumi.packing._core`. Everything numpy- or I/O-bound (streaming, reports,
masking) stays in pure Python. `takumi/packing/_core.pyi` is the hand-written
type stub that gives editors and mypy signatures for the compiled module — keep
it in sync when the bindings change.

Build (scikit-build-core → CMake → clang++ → `_core.*.so` in `takumi/packing/`):

```bash
python -m venv .venv
.venv/bin/pip install nanobind scikit-build-core cmake ninja numpy pytest torch

# editable install — compiles the C++ and links the package into your tree
.venv/bin/pip install --no-build-isolation -e .

.venv/bin/pytest
```

`torch` is in that list only because the masking tests exercise it. Packing
needs neither torch nor JAX: `takumi/__init__.py` imports `takumi.masking`
lazily, so `from takumi.packing import pack` works with neither installed.

Rebuilding after edits:

| Edited                              | Run                                                                       |
|-------------------------------------|---------------------------------------------------------------------------|
| a `.py` file                        | nothing — the editable install points at your tree                        |
| a `.cpp` file                       | `.venv/bin/pip install --no-build-isolation -e .` (incremental recompile)  |
| `CMakeLists.txt` / `pyproject.toml` | `rm -rf build && .venv/bin/pip install --no-build-isolation -e .`         |
