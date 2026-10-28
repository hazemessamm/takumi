// packing.cpp — C++ port of the bin-packing core, exposed to Python via
// nanobind. This file compiles into the module `takumi.packing._core`.
//
// The core is First-Fit-Decreasing (`pack_first_fit`), Worst-Fit-Decreasing
// (`pack_worst_fit`), and the histogram-based packers (`group_by_size` /
// `pack_histogram`). The bindings block at the bottom marks where each is
// exposed to Python.
//
// nanobind cheat-sheet used below:
//   nb::list / nb::int_   -> Python objects built from C++
//   NB_MODULE(name, m)    -> defines the extension module `name`
//   m.def("py_name", &fn) -> exposes a C++ function to Python
//   nb::arg("x") = default-> names an argument and gives it a default

#include <algorithm>
#include <cstdint>
#include <iterator>  // std::make_move_iterator
#include <optional>
#include <stdexcept>
#include <string>   // std::to_string — pulled in transitively by libstdc++/libc++, but not by MSVC
#include <vector>
#include <random>
#include <numeric>
#include <tuple>
#include <utility>
#include <map>
#include <ranges>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // enables std::optional <-> None|value casts
#include <nanobind/stl/vector.h>    // enables automatic std::vector <-> list casts
#include <nanobind/stl/unordered_map.h>  // enables automatic std::unordered_map <-> dict casts
#include <nanobind/stl/map.h>  // enables automatic std::map -> dict casts (sorted keys)
#include <nanobind/stl/tuple.h>  // enables automatic std::tuple <-> tuple casts


namespace nb = nanobind;

namespace {

// Reject sequences that cannot fit a bin even on their own. Without this the
// concrete packers happily open a bin with negative remaining capacity and
// return a packing that violates max_length; pack_histogram already refuses
// the same input, so this keeps the two families on one contract.
// std::invalid_argument surfaces in Python as ValueError.
void require_fits(
    const std::vector<std::int64_t>& lengths,
    std::int64_t max_length) {
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] > max_length) {
            throw std::invalid_argument(
                "Length " + std::to_string(lengths[i]) + " at index " +
                std::to_string(i) + " exceeds max_length " +
                std::to_string(max_length));
        }
    }
}

// Return indices into `lengths`, sorted by (length / tolerance) descending.
// Two code paths: deterministic (seed=None) and shuffled (seed set).
//
// seed is None (deterministic): sort by bucket descending, ties broken by
// original index via std::stable_sort.
//   NOTE on fidelity: the original uses np.argsort(-buckets), and numpy's
//   default argsort is *quicksort* — UNSTABLE for ties (the "deterministic
//   stable sort" in its docstring is a misnomer). So for inputs with
//   equal-length sequences, this port assigns those equal items to bins in a
//   different order than the original. The packings are equivalent in every
//   quality metric (identical bin count and identical multiset of per-bin
//   token sums; verified over thousands of random inputs) — only *which*
//   equal-length index lands in which bin changes. Stable sort is deliberate:
//   it is reproducible across platforms and numpy versions, which quicksort
//   tie-order is not.
//
// seed is set (shuffled): give each item a random tie-breaker and sort by
// (bucket descending, tie_breaker ascending, index ascending) — the same
// ordering numpy gets from np.lexsort((tie_breakers, -buckets)). Same seed,
// same order on every platform: see the comments in the code below.
//   NOTE on fidelity: the tie-breakers come from std::mt19937_64, NOT numpy's
//   PCG64. So the output is reproducible given a seed but is NOT identical to
//   the original numpy `seed` output — a deliberate choice. Matching numpy
//   bit-for-bit would require reimplementing PCG64.
std::vector<std::size_t> order_by_size(
    const std::vector<std::int64_t>& lengths,
    std::int64_t tolerance,
    std::optional<std::uint64_t> seed = std::nullopt) {
    // The bucket key below divides by `tolerance`. Integer division by zero is
    // undefined behaviour in C++ — it yields 0 on arm64 (silently collapsing
    // every item into one bucket) but raises SIGFPE on x86-64, killing the
    // interpreter. A negative divisor is no better: it inverts the ordering,
    // which is meaningless for a bucket *width*.
    if (tolerance <= 0) {
        throw std::invalid_argument(
            "tolerance must be >= 1, got " + std::to_string(tolerance));
    }

    const std::size_t n = lengths.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::vector<int64_t> buckets(n);
    for (std::size_t i = 0; i < n; ++i) {
        buckets[i] = lengths[i] / tolerance;
    }

    if (seed.has_value()) {
        // The 64-bit engine, so every bit of the seed counts: std::mt19937
        // keeps only the low 32, making seeds 2**32 apart identical.
        std::mt19937_64 gen(seed.value());

        // Raw engine output, not a std:: distribution: the standard fixes
        // mt19937_64's output bit for bit, but leaves each distribution's
        // algorithm to the library, so libstdc++, libc++ and MSVC would
        // shuffle the same seed differently.
        std::vector<std::uint64_t> tie_breakers(n);

        for (std::size_t i = 0; i < n; ++i) {
            tie_breakers[i] = gen();
        }

        // The index settles equal tie-breakers, so no two items compare equal
        // and std::sort, whose order for equal items differs by library, has
        // no choice left to make.
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return std::make_tuple(-buckets[a], tie_breakers[a], a) < std::make_tuple(-buckets[b], tie_breakers[b], b);
        });
    }
    else {
        // Bucket key = length / tolerance. Sort by bucket descending; on ties keep
        // the smaller original index first (stable, like np.argsort).
        std::stable_sort(order.begin(), order.end(),
                        [&](std::size_t a, std::size_t b) {
                            return (lengths[a] / tolerance) > (lengths[b] / tolerance);
                        });
    }
    return order;
}

// First-Fit-Decreasing bin packing.
//
// Sort sequences longest
// first, then place each into the first (oldest) open bin that still has room;
// open a fresh bin when none fits. Returns a list of bins, each a list of
// original indices into `lengths`.
std::vector<std::vector<std::int64_t>> pack_first_fit(
    const std::vector<std::int64_t>& lengths,
    std::int64_t max_length,
    std::int64_t tolerance,
    std::optional<std::uint64_t> seed) {

    require_fits(lengths, max_length);
    const std::vector<std::size_t> order = order_by_size(lengths, tolerance, seed);

    std::vector<std::vector<std::int64_t>> bins;
    std::vector<std::int64_t> bin_remaining;  // parallel to `bins`

    for (const std::size_t orig_idx : order) {
        const std::int64_t seq_len = lengths[orig_idx];
        bool placed = false;

        // First fit: linear scan from the oldest bin to the newest.
        for (std::size_t b = 0; b < bins.size(); ++b) {
            if (bin_remaining[b] >= seq_len) {
                bins[b].push_back(static_cast<std::int64_t>(orig_idx));
                bin_remaining[b] -= seq_len;
                placed = true;
                break;
            }
        }

        if (!placed) {
            bins.push_back({static_cast<std::int64_t>(orig_idx)});
            bin_remaining.push_back(max_length - seq_len);
        }
    }

    return bins;
}


std::vector<std::vector<std::int64_t>> pack_worst_fit(
    const std::vector<std::int64_t>& lengths,
    std::int64_t max_length,
    std::int64_t tolerance,
    std::optional<std::uint64_t> seed) {

    require_fits(lengths, max_length);
    const std::vector<std::size_t> order = order_by_size(lengths, tolerance, seed);
    std::vector<std::vector<std::int64_t>> bins;
    // Max-heap on (remaining_capacity, bin_idx): front() is the loosest bin (WFD).
    // NOTE on fidelity: on equal remaining capacity this tie-breaks toward the
    // LARGER bin_idx, whereas the original Python heapq (min-heap on
    // (-remaining, bin_idx)) tie-breaks toward the SMALLER bin_idx. Both are
    // valid WFD and quality-equivalent — identical bin count and identical
    // multiset of per-bin fill (verified over thousands of random inputs); only
    // *which* index lands in an otherwise-identical bin differs. Not matched to
    // Python on purpose, consistent with the sort and RNG choices above.
    std::vector<std::tuple<std::int64_t, std::size_t>> heap;  // (remaining_capacity, bin_idx)

    for (std::size_t orig_idx : order) {
        std::int64_t seq_len = lengths[orig_idx];

        if (!heap.empty() && std::get<0>(heap.front()) >= seq_len) {
            std::pop_heap(heap.begin(), heap.end());
            auto [remaining_capacity, bin_idx] = heap.back();
            heap.pop_back();
            bins[bin_idx].push_back(static_cast<std::int64_t>(orig_idx));
            remaining_capacity -= seq_len;
            heap.push_back(std::make_tuple(remaining_capacity, bin_idx));
            std::push_heap(heap.begin(), heap.end());
        }
        else {
            std::size_t bin_idx = bins.size();
            bins.push_back({static_cast<std::int64_t>(orig_idx)});
            heap.emplace_back(max_length - seq_len, bin_idx);
            std::push_heap(heap.begin(), heap.end());
        }
    }
    return bins;
    }


    // Counts in hash maps (O(1) per sequence), then returns std::maps so Python
    // sees ascending keys everywhere: unordered_map iteration order differs
    // between libstdc++, libc++ and MSVC. The copy is O(k log k) over the k
    // distinct lengths, negligible next to the O(n) pass.
    std::tuple<std::map<std::int64_t, std::int64_t>, std::map<std::int64_t, std::vector<std::int64_t>>> group_by_size(
        std::vector<std::int64_t>& lengths, std::optional<std::int64_t> max_length = std::nullopt) {
            if (max_length.has_value()) {
                for (std::size_t i = 0; i < lengths.size(); ++i) {
                    lengths[i] = std::min(lengths[i], max_length.value());
                }
            }

            std::unordered_map<std::int64_t, std::int64_t> counts;
            for (auto length : lengths) {
                counts[length] += 1;
            }

            std::unordered_map<std::int64_t, std::vector<std::int64_t>> length_to_indices;
            for (std::size_t i = 0; i < lengths.size(); ++i) {
                length_to_indices[lengths[i]].push_back(static_cast<std::int64_t>(i));
            }

            return std::make_tuple(
                std::map<std::int64_t, std::int64_t>(counts.begin(), counts.end()),
                std::map<std::int64_t, std::vector<std::int64_t>>(
                    std::make_move_iterator(length_to_indices.begin()),
                    std::make_move_iterator(length_to_indices.end())));
        }


    // Collapse a list of per-bin length compositions into a Python dict mapping
    // each distinct composition (as a tuple of lengths) to the number of bins
    // that share it -- the {template: count} shape both histogram packers
    // return. Routing through std::map gives the keys a stable, sorted order.
    nb::dict templates_from_compositions(
        const std::vector<std::vector<std::int64_t>>& bin_compositions) {
        std::map<std::vector<std::int64_t>, std::int64_t> grouped_results;
        for (const auto& bin : bin_compositions) {
            grouped_results[bin] += 1;
        }

        nb::dict result;
        for (const auto& [bin, count] : grouped_results) {
            // nb::cast(bin) turns the C++ vector into a Python *list*
            // (via nanobind/stl/vector.h). A list is unhashable, so it
            // can't be a dict key; PySequence_Tuple copies it into a
            // hashable tuple. nb::steal adopts the new reference it returns.
            nb::object as_list = nb::cast(bin);
            nb::tuple key = nb::steal<nb::tuple>(PySequence_Tuple(as_list.ptr()));
            result[key] = count;
        }
        return result;
    }


    nb::dict pack_histogram(
        const std::unordered_map<std::int64_t, std::int64_t>& counts,
        std::int64_t max_length) {

            for (const auto& [length, count] : counts) {
                if (length > max_length) {
                    throw std::invalid_argument("Length " + std::to_string(length) + " exceeds max_length " + std::to_string(max_length));
                }
            }

            std::vector<std::pair<std::int64_t, std::size_t>> heap;
            std::vector<std::vector<std::int64_t>> bin_compositions;

            auto lengths = counts | std::views::keys | std::ranges::to<std::vector<std::int64_t>>();

            std::sort(lengths.begin(), lengths.end(), std::greater<std::int64_t>());


            for (const auto& length : lengths) {
                for (std::int64_t i = 0; i < counts.at(length); ++i) {
                    if (!heap.empty() && heap.front().first >= length) {
                        auto [remaining_capacity, bin_idx] = heap.front();
                        std::pop_heap(heap.begin(), heap.end());
                        heap.pop_back();
                        bin_compositions[bin_idx].push_back(length);
                        auto new_remaining_capacity = remaining_capacity - length;
                        heap.emplace_back(new_remaining_capacity, bin_idx);
                        std::push_heap(heap.begin(), heap.end());
                    }
                    else {
                        std::size_t bin_idx = bin_compositions.size();
                        bin_compositions.push_back({length});
                        heap.emplace_back(max_length - length, bin_idx);
                        std::push_heap(heap.begin(), heap.end());
                    }
                }
            }

            return templates_from_compositions(bin_compositions);
    }


    // Batched worst-fit-decreasing packing on a length histogram.
    //
    // Same inputs and output shape as pack_histogram, but instead of placing
    // sequences one at a time it processes a whole run of equal-length items per
    // heap step: for each distinct length (descending) it max-fills the loosest
    // bins first (phase 1), then opens new max-filled bins for whatever is left
    // (phase 2). Collapsing equal-length runs into single operations makes it
    // faster than pack_histogram by a factor that grows with how many items fit
    // per bin.
    //
    // NOTE on fidelity: this differs from the per-item pack_histogram in two
    // deliberate ways, matching the original seqpack batched packer.
    //   (1) It max-fills one bin before moving on, rather than spreading
    //       equal-length items across bins one at a time. Bin count matches
    //       pack_histogram on realistic length distributions and differs by at
    //       most one bin on adversarial inputs; the templates tend to group more
    //       identical lengths into the same bin.
    //   (2) Like the other heap packers here, ties on remaining capacity break
    //       toward the LARGER bin_idx (std::push_heap) rather than the smaller
    //       one (Python heapq). Quality-equivalent, not assignment-identical.
    nb::dict pack_histogram_batched(
        const std::unordered_map<std::int64_t, std::int64_t>& counts,
        std::int64_t max_length) {

        for (const auto& [length, count] : counts) {
            if (length > max_length) {
                throw std::invalid_argument("Length " + std::to_string(length) + " exceeds max_length " + std::to_string(max_length));
            }
        }

        std::vector<std::pair<std::int64_t, std::size_t>> heap;  // max-heap on (remaining_capacity, bin_idx)
        std::vector<std::vector<std::int64_t>> bin_compositions;

        auto lengths = counts | std::views::keys | std::ranges::to<std::vector<std::int64_t>>();
        std::sort(lengths.begin(), lengths.end(), std::greater<std::int64_t>());

        for (const auto& length : lengths) {
            std::int64_t remaining_count = counts.at(length);

            // Phase 1: max-fill existing bins, loosest (worst-fit) first. The
            // while-guard stops once no open bin can hold even one more item of
            // this length, since each popped bin is filled down to < length.
            while (remaining_count > 0 && !heap.empty() && heap.front().first >= length) {
                std::pop_heap(heap.begin(), heap.end());
                auto [bin_remaining, bin_idx] = heap.back();
                heap.pop_back();

                std::int64_t k = std::min(remaining_count, bin_remaining / length);
                bin_compositions[bin_idx].insert(
                    bin_compositions[bin_idx].end(), static_cast<std::size_t>(k), length);
                bin_remaining -= k * length;

                heap.emplace_back(bin_remaining, bin_idx);
                std::push_heap(heap.begin(), heap.end());
                remaining_count -= k;
            }

            // Phase 2: open new bins for the remainder, each max-filled.
            std::int64_t items_per_new_bin = max_length / length;
            while (remaining_count > 0) {
                std::int64_t k = std::min(remaining_count, items_per_new_bin);
                std::size_t bin_idx = bin_compositions.size();
                bin_compositions.emplace_back(static_cast<std::size_t>(k), length);
                heap.emplace_back(max_length - k * length, bin_idx);
                std::push_heap(heap.begin(), heap.end());
                remaining_count -= k;
            }
        }

        return templates_from_compositions(bin_compositions);
    }

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "C++/nanobind core for takumi.packing (bin-packing hot paths).";

    m.def("pack_first_fit", &pack_first_fit,
          nb::arg("lengths"),
          nb::arg("max_length"),
          nb::arg("tolerance") = 1,
          nb::arg("seed") = nb::none(),
          R"doc(First-Fit-Decreasing bin packing.

Sorts sequences longest first, then places each into the first (oldest) open
bin that still has room, opening a fresh bin when none fits.

Args:
    lengths (Sequence[int], length N): Sequence lengths, each at most
        ``max_length``.
    max_length (int): Bin capacity in tokens.
    tolerance (int): Bucket width for the longest-first sort, ``>= 1``;
        lengths in the same bucket count as ties.
    seed (int | None): Shuffles ties reproducibly on every platform;
        any integer in ``[0, 2**64)``, like NumPy's. ``None`` breaks ties by
        original index.

Returns:
    list[list[int]]: Bins, each a list of indices into ``lengths``.

Raises:
    ValueError: if ``tolerance`` is below 1, or any length exceeds
        ``max_length``.

Example:
    >>> pack_first_fit([7, 6, 5, 4, 3, 2], max_length=10)
    [[0, 4], [1, 3], [2, 5]]
    >>> a = pack_first_fit([7, 6, 5, 4, 3, 2], 10, tolerance=2, seed=42)
    >>> a == pack_first_fit([7, 6, 5, 4, 3, 2], 10, tolerance=2, seed=42)
    True
)doc");

    m.def("pack_worst_fit", &pack_worst_fit,
          nb::arg("lengths"),
          nb::arg("max_length"),
          nb::arg("tolerance") = 1,
          nb::arg("seed") = nb::none(),
          R"doc(Heap-based Worst-Fit-Decreasing bin packing.

Sorts sequences longest first, then places each into the *loosest* open bin
that still has room — the bin with the most remaining capacity — rather than
the first one that fits. A max-heap keyed on remaining capacity makes that
choice O(log n) per sequence.

Args:
    lengths (Sequence[int], length N): Sequence lengths, each at most
        ``max_length``.
    max_length (int): Bin capacity in tokens.
    tolerance (int): Bucket width for the longest-first sort, ``>= 1``;
        lengths in the same bucket count as ties.
    seed (int | None): Shuffles ties reproducibly on every platform;
        any integer in ``[0, 2**64)``, like NumPy's. ``None`` breaks ties by
        original index.

Returns:
    list[list[int]]: Bins, each a list of indices into ``lengths``.

Raises:
    ValueError: if ``tolerance`` is below 1, or any length exceeds
        ``max_length``.

Example:
    >>> pack_worst_fit([7, 6, 5, 4, 3, 2], max_length=10)
    [[0, 5], [1, 4], [2, 3]]
)doc");

    m.def("group_by_size", &group_by_size,
          nb::arg("lengths"),
          nb::arg("max_length") = nb::none(),
          R"doc(Compute a histogram of sequence lengths and a mapping from lengths to indices.

Args:
    lengths (Sequence[int], length N): Sequence lengths.
    max_length (int | None): If given, lengths above it are clipped to it.

Returns:
    tuple[dict[int, int], dict[int, list[int]]]: ``(counts,
    length_to_indices)``: how many sequences have each length, and their
    indices in ascending order. Both dicts list lengths in ascending order
    on every platform.

Example:
    >>> counts, pools = group_by_size([3, 5, 3, 9], max_length=8)
    >>> counts  # 9 was clipped to 8
    {3: 2, 5: 1, 8: 1}
    >>> pools
    {3: [0, 2], 5: [1], 8: [3]}
)doc");

    m.def("pack_histogram", &pack_histogram,
          nb::arg("counts"),
          nb::arg("max_length"),
          R"doc(Pack a histogram of sequence lengths into bins.

Worst-fit-decreasing over lengths rather than individual sequences, so the
result is a reusable blueprint of bin layouts.

Args:
    counts (Mapping[int, int]): Length to how many sequences have it; each
        length at most ``max_length``.
    max_length (int): Bin capacity in tokens.

Returns:
    dict[tuple[int, ...], int]: Bin layout (the lengths sharing a bin,
    longest first) to how many bins use it, in sorted key order.

Raises:
    ValueError: if any length exceeds ``max_length``.

Example:
    >>> pack_histogram({7: 1, 6: 1, 5: 1, 4: 1, 3: 1, 2: 1}, max_length=10)
    {(5, 4): 1, (6, 3): 1, (7, 2): 1}
)doc");

    m.def("pack_histogram_batched", &pack_histogram_batched,
          nb::arg("counts"),
          nb::arg("max_length"),
          R"doc(Batched worst-fit-decreasing packing of a length histogram.

Same inputs and output as pack_histogram, but fills a whole run of equal-length
items per heap operation instead of one at a time, which is faster when many
items fit per bin. The packing is quality-equivalent to pack_histogram (bin
count matches on realistic inputs, differs by at most one bin on adversarial
ones); its templates tend to group more identical lengths per bin.

Args:
    counts (Mapping[int, int]): Length to how many sequences have it; each
        length at most ``max_length``.
    max_length (int): Bin capacity in tokens.

Returns:
    dict[tuple[int, ...], int]: Bin layout (the lengths sharing a bin,
    longest first) to how many bins use it, in sorted key order.

Raises:
    ValueError: if any length exceeds ``max_length``.

Example:
    >>> pack_histogram_batched({2: 10, 3: 4}, max_length=8)
    {(2, 2, 2, 2): 2, (3, 3, 2): 2}
)doc");
}
