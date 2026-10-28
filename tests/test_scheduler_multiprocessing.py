"""Cross-process behaviour of the curriculum schedulers.

``SharedCounter`` wraps a ``multiprocessing.Value`` so a curriculum can be
stepped from inside DataLoader workers and have the parent observe it. That
guarantee is load-bearing and start-method dependent — fork inherits the
shared memory, while spawn and forkserver re-create the handle through
``ForkingPickler`` — so it is pinned here for every start method the platform
offers, with one and with several concurrent workers.

Note the deliberate *non*-guarantee in ``test_plain_pickle_is_refused``: a bare
``pickle.dumps`` of a scheduler raises, and that is multiprocessing's guardrail
working as designed, not a defect. ``Synchronized.__reduce__`` permits itself
only inside a genuine spawning context. Adding ``__getstate__``/``__setstate__``
to make plain pickling "work" would silently swap real sharing for per-worker
snapshots whose steps never reach the parent, so the refusal is asserted to
keep that change from looking like an improvement.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle

import pytest

from takumi.masking.scheduler.common import SharedCounter
from takumi.masking.scheduler.linear import LinearScheduler

# fork is absent on Windows; forkserver is absent on some platforms. Only
# exercise what this platform actually supports.
START_METHODS = [
    method
    for method in ("fork", "spawn", "forkserver")
    if method in mp.get_all_start_methods()
]

# Child targets must be importable module-level functions: spawn and forkserver
# pickle them by qualified name and re-import them in the child.


def _step_counter(counter: SharedCounter, times: int) -> None:
    for _ in range(times):
        counter.step()


def _step_scheduler(scheduler: LinearScheduler, times: int) -> None:
    for _ in range(times):
        scheduler.step()


def _load_scheduler_state(scheduler: LinearScheduler, step: int) -> None:
    scheduler.load_state_dict({"step": step})


def _run_children(method, target, arg, n_procs, times):
    """Start ``n_procs`` children, each calling ``target(arg, times)``."""
    ctx = mp.get_context(method)
    procs = [
        ctx.Process(target=target, args=(arg, times)) for _ in range(n_procs)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
    for proc in procs:
        assert proc.exitcode == 0, f"child failed with exitcode {proc.exitcode}"


@pytest.mark.parametrize("method", START_METHODS)
@pytest.mark.parametrize("n_procs", [1, 2])
def test_counter_is_shared_across_processes(method, n_procs):
    """Steps taken in children are visible to the parent under every method."""
    counter = SharedCounter(initial_value=0, final_value=1000)
    _run_children(method, _step_counter, counter, n_procs, times=5)
    # step() holds the lock, so concurrent increments cannot be lost.
    assert counter.get() == 5 * n_procs


@pytest.mark.parametrize("method", START_METHODS)
@pytest.mark.parametrize("n_procs", [1, 2])
def test_children_start_from_the_parents_value(method, n_procs):
    """A child inherits the counter's current value, not its initial one."""
    counter = SharedCounter(initial_value=0, final_value=1000)
    for _ in range(3):
        counter.step()
    _run_children(method, _step_counter, counter, n_procs, times=2)
    assert counter.get() == 3 + 2 * n_procs


@pytest.mark.parametrize("method", START_METHODS)
def test_final_value_clamp_holds_across_processes(method):
    """The ceiling is enforced in the shared value, not per-process."""
    counter = SharedCounter(initial_value=0, final_value=3)
    _run_children(method, _step_counter, counter, n_procs=2, times=5)
    assert counter.get() == 3


@pytest.mark.parametrize("method", START_METHODS)
@pytest.mark.parametrize("n_procs", [1, 2])
def test_scheduler_curriculum_advances_from_children(method, n_procs):
    """The whole Scheduler — not just the bare counter — survives the trip."""
    scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=100)
    before = scheduler.sample()
    _run_children(method, _step_scheduler, scheduler, n_procs, times=10)

    assert scheduler.current_step.get() == 10 * n_procs
    # Weights interpolate with the shared step count, so the parent's view of
    # the curriculum actually moved.
    after = scheduler.sample()
    assert after != before
    assert after == pytest.approx([1.0 - 0.1 * n_procs, 0.1 * n_procs])


@pytest.mark.parametrize("method", START_METHODS)
def test_reset_from_parent_is_seen_by_later_children(method):
    scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=100)
    _run_children(method, _step_scheduler, scheduler, n_procs=1, times=7)
    assert scheduler.current_step.get() == 7

    scheduler.reset()
    assert scheduler.current_step.get() == 0

    _run_children(method, _step_scheduler, scheduler, n_procs=1, times=2)
    assert scheduler.current_step.get() == 2


@pytest.mark.parametrize("method", START_METHODS)
@pytest.mark.parametrize("n_procs", [1, 2])
def test_children_continue_from_a_loaded_state(method, n_procs):
    """Resuming in the parent before workers start: they pick up from there."""
    scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=100)
    scheduler.load_state_dict({"step": 40})
    _run_children(method, _step_scheduler, scheduler, n_procs, times=3)
    assert scheduler.state_dict() == {"step": 40 + 3 * n_procs}


@pytest.mark.parametrize("method", START_METHODS)
def test_state_loaded_in_a_child_reaches_the_parent(method):
    """load_state_dict writes the shared value, exactly as step() does."""
    scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=100)
    _run_children(method, _load_scheduler_state, scheduler, n_procs=1, times=25)
    assert scheduler.state_dict() == {"step": 25}


def test_plain_pickle_is_refused():
    """Bare pickling raises — see the module docstring before "fixing" this.

    Real process creation goes through ForkingPickler inside a spawning
    context, which is why the tests above pass while this one does not.
    """
    scheduler = LinearScheduler([1.0, 0.0], [0.0, 1.0], total_steps=100)
    with pytest.raises(RuntimeError, match="through inheritance"):
        pickle.dumps(scheduler)
