"""packing — sequence bin-packing.

The performance-critical packers live in the compiled extension
``takumi.packing._core`` (see ``src/packing/packing.cpp``). Pure-Python helpers
that are I/O- or numpy-bound (stats, streaming) stay here in Python. This
``__init__`` re-exports both so callers see one flat namespace.

``pack(lengths, max_length, algorithm=...)`` is the main entry point; the raw
per-algorithm functions remain available for callers that want them directly.
"""

from takumi.packing._core import pack_first_fit
from takumi.packing._core import pack_worst_fit
from takumi.packing._core import group_by_size
from takumi.packing._core import pack_histogram
from takumi.packing._core import pack_histogram_batched
from takumi.packing.packing_report import PackingReport
from takumi.packing.stream import stream_packed_bins
from takumi.packing.packing import Bins
from takumi.packing.packing import pack
from takumi.packing.packing import PackingPlan

__all__ = [
    "pack",
    "Bins",
    "PackingPlan",
    "PackingReport",
    "pack_first_fit",
    "pack_worst_fit",
    "group_by_size",
    "pack_histogram",
    "pack_histogram_batched",
    "stream_packed_bins",
]
