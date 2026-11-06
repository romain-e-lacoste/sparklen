# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Ownership of the arrays passed to and returned by the C++ core."""

import gc
import tracemalloc

import numpy as np
import pytest
from hawkes_numpy_reference import tiny_paths

from sparklen.hawkes.model import (
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)
from sparklen.hawkes.model.build.hawkes_model import (
    ModelHawkesExpLogLikelihood as CppModelHawkesExpLogLikelihood,
    ModelHawkesExpLogLikelihoodSingle as CppModelHawkesExpLogLikelihoodSingle,
)


def returned_arrays():
    theta = np.full((2, 3), 0.3)
    least_squares = ModelHawkesExpLeastSquares(decay=1.0)
    least_squares.set_data(tiny_paths(), end_time=5.0)
    log_likelihood = ModelHawkesExpLogLikelihood(decay=1.0)
    log_likelihood.set_data(tiny_paths(), end_time=5.0)
    return [least_squares.grad(theta), least_squares.hessian(),
            log_likelihood.grad(theta)]


def test_returned_arrays_are_not_copied():
    """Check that the C++ core passes its arrays to NumPy without a copy:
    NumPy allocates no memory for them."""
    returned_arrays()  # first call: NumPy may allocate caches once
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        numpy_memory = [tracemalloc.DomainFilter(True, np.lib.tracemalloc_domain)]
        before = tracemalloc.take_snapshot().filter_traces(numpy_memory)
        arrays = [returned_arrays() for _ in range(5)]
        after = tracemalloc.take_snapshot().filter_traces(numpy_memory)
    finally:
        if started:
            tracemalloc.stop()
    assert len(arrays) == 5
    assert sum(stat.size_diff for stat in after.compare_to(before, "filename")) == 0


def test_returned_arrays_are_kept_alive_by_their_owner():
    """Check that NumPy does not own the returned buffers, which C++ frees,
    that an owner object keeps them alive, and that they are writeable and
    C-contiguous."""
    for array in returned_arrays():
        assert not array.flags.owndata
        assert array.base is not None
        assert array.flags.c_contiguous
        assert array.flags.writeable


def test_returned_arrays_outlive_the_model():
    """Check that returned arrays keep their values after the model is freed
    and its memory reused."""
    arrays = returned_arrays()
    expected = [array.copy() for array in arrays]
    gc.collect()
    for _ in range(200):
        returned_arrays()  # allocate and free many buffers
    for array, values in zip(arrays, expected):
        np.testing.assert_array_equal(array, values)


def test_each_call_returns_its_own_buffer():
    """Check that two calls return arrays that do not share memory, so that
    writing into one leaves the other unchanged."""
    model = ModelHawkesExpLeastSquares(decay=1.0)
    model.set_data(tiny_paths(), end_time=5.0)
    theta = np.full((2, 3), 0.3)
    first, second = model.grad(theta), model.grad(theta)
    assert not np.shares_memory(first, second)
    expected = second.copy()
    first[...] = 0.0
    np.testing.assert_array_equal(second, expected)


@pytest.mark.parametrize("cls", [ModelHawkesExpLeastSquares, ModelHawkesExpLogLikelihood])
def test_many_gradient_calls_are_stable(cls):
    """Check that many repeated gradient calls return identical results."""
    model = cls(decay=1.0)
    model.set_data(tiny_paths(), end_time=5.0)
    theta = np.full((2, 3), 0.3)
    first = model.grad(theta)
    for _ in range(2000):
        grad = model.grad(theta)
    np.testing.assert_array_equal(grad, first)


def large_path():
    # 600 events per component: more than NumPy's small-buffer cache, so a
    # freed buffer goes back to the allocator, which reuses it
    rng = np.random.default_rng(0)
    return [np.sort(rng.uniform(0.0, 10.0, 600)) for _ in range(3)]


@pytest.mark.parametrize("built_on_the_fly", ["generator of paths", "generator of events"])
def test_paths_built_on_the_fly_stay_alive_during_the_call(built_on_the_fly):
    """Check that the C++ core reads the right event times when the paths are
    passed as generators of new arrays, which only the call keeps alive."""
    path, theta = large_path(), np.full((3, 4), 0.1)
    expected = CppModelHawkesExpLogLikelihood(1, 3).compute_averaged_grad(
        [path], 10.0, 1.0, theta, True)
    if built_on_the_fly == "generator of paths":
        paths = ([events.copy() for events in p] for p in [path])
    else:
        paths = [(events.copy() for events in path)]
    grad = CppModelHawkesExpLogLikelihood(1, 3).compute_averaged_grad(
        paths, 10.0, 1.0, theta, True)
    np.testing.assert_array_equal(grad, expected)


def test_events_built_on_the_fly_stay_alive_during_the_call():
    """Check the same for one path passed as a generator of new arrays."""
    path, theta = large_path(), np.full((3, 4), 0.1)
    expected = CppModelHawkesExpLogLikelihoodSingle(3).compute_grad(path, 10.0, 1.0, theta, True)
    grad = CppModelHawkesExpLogLikelihoodSingle(3).compute_grad(
        (events.copy() for events in path), 10.0, 1.0, theta, True)
    np.testing.assert_array_equal(grad, expected)
