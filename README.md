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
)

diffusion = DiffusionMasker(
    total_timesteps=1000,
    mask_token_id=3,
    special_token_ids=[0, 1, 2, 3],
    schedule="cosine",  # or "linear", "sqrt", or a callable
)
result = diffusion(x, seed=10, timestep=500)
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
  `SampledRate(sampler)`, `ScheduleRate(total_timesteps, schedule)`
- selectors — `Bernoulli()`, `FocusBiased(strategy, strength)`, `Spans(...)`
- schemes — `Mixed(mask_prob, mutate_prob, mask_token_id, valid_token_ids)`,
  `MaskOnly(mask_token_id)`

Per-call options:

```python
result = corruptor(
    x,
    seed=10,                  # bit-exact reproducible call
    excluded_ids=[0, 1],      # overrides special_token_ids for this call
    focus=focus_mask,         # (B, L) bool mask to bias selection toward
    timestep=500,             # required by ScheduleRate, ignored otherwise
)
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
