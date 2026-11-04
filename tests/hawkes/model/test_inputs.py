# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Input validation and memory layout of arrays passed to the C++ core."""

import numpy as np
import pytest
from hawkes_numpy_reference import tiny_paths

from sparklen.hawkes.model import (
    ModelHawkesExpClassification,
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)

END_TIME = 5.0
MODELS = [ModelHawkesExpLeastSquares, ModelHawkesExpLogLikelihood]


@pytest.fixture
def data():
    return tiny_paths(n_paths=3, n_components=2, end_time=END_TIME)


@pytest.fixture
def theta():
    return np.array([[0.4, 0.3, 0.1],
                     [0.2, 0.05, 0.25]])


def make_model(cls, data):
    model = cls(decay=1.5)
    model.set_data(data, end_time=END_TIME)
    return model


@pytest.mark.parametrize("cls", MODELS)
def test_fortran_ordered_theta_gives_the_same_results(cls, data, theta):
    """Check that a Fortran-ordered theta gives the same loss and gradient
    as a C-ordered one."""
    model = make_model(cls, data)
    fortran = np.asfortranarray(theta)
    assert model.loss(fortran) == model.loss(theta)
    np.testing.assert_array_equal(model.grad(fortran), model.grad(theta))


@pytest.mark.parametrize("cls", MODELS)
def test_integer_theta_is_converted(cls, data):
    """Check that an integer theta gives the same loss as its float64 copy."""
    model = make_model(cls, data)
    theta = np.ones((2, 3), dtype=int)
    assert model.loss(theta) == model.loss(theta.astype(float))


@pytest.mark.parametrize("cls", MODELS)
def test_strided_or_list_events_give_the_same_results(cls, data, theta):
    """Check that event times given as strided views or lists give the same loss."""
    strided = []
    for path in data:
        new_path = []
        for events in path:
            buffer = np.zeros(2 * len(events))
            buffer[::2] = events
            new_path.append(buffer[::2])  # a non-contiguous view
        strided.append(new_path)
    as_lists = [[events.tolist() for events in path] for path in data]
    expected = make_model(cls, data).loss(theta)
    assert make_model(cls, strided).loss(theta) == expected
    assert make_model(cls, as_lists).loss(theta) == expected


@pytest.mark.parametrize("bad_events, message", [
    (np.array([1.0, 0.5, 2.0]), "sorted"),
    (np.array([0.5, np.nan]), "NaN or infinite"),
    (np.array([-1.0, 2.0]), "negative"),
    (np.array([1.0, 6.0]), "exceeds"),
    (np.ones((2, 2)), "one-dimensional"),
])
@pytest.mark.parametrize("cls", MODELS)
def test_invalid_event_times_are_rejected(cls, bad_events, message):
    """Check that unsorted, non-finite, negative, later than T or 2-D event times
    raise a ValueError."""
    model = cls(decay=1.0)
    with pytest.raises(ValueError, match=message):
        model.set_data([[bad_events, np.array([1.0])]], end_time=END_TIME)


@pytest.mark.parametrize("end_time", [0.0, -1.0, np.inf])
def test_invalid_end_time_is_rejected(end_time):
    """Check that a zero, negative or infinite end time raises a ValueError."""
    model = ModelHawkesExpLeastSquares(decay=1.0)
    with pytest.raises(ValueError, match="upper bound"):
        model.set_data([[np.array([0.5])]], end_time=end_time)


def test_end_time_cannot_be_inferred_without_events():
    """Check that a missing end time raises a ValueError when there is no event."""
    model = ModelHawkesExpLeastSquares(decay=1.0)
    with pytest.raises(ValueError, match="cannot be inferred"):
        model.set_data([[np.array([])]])


def test_classification_labels_are_checked(data):
    """Check that negative labels, or labels of the wrong length, raise a ValueError."""
    model = ModelHawkesExpClassification(decay=1.5, weights=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="non-negative integers"):
        model.set_data(data, np.array([0, -1, 1]), END_TIME)
    with pytest.raises(ValueError, match="same length"):
        model.set_data(data, np.array([0, 1]), END_TIME)


def test_the_cpp_core_rejects_non_contiguous_arrays(theta):
    """Check that the C++ core itself rejects non-contiguous event times and theta."""
    from sparklen.hawkes.model.build.hawkes_model import (
        ModelHawkesExpLogLikelihoodSingle as CppModelHawkesExpLogLikelihoodSingle,
    )

    path = [np.array([0.5, 1.0, 2.0]), np.zeros(6)[::2]]
    with pytest.raises(ValueError, match="C-contiguous"):
        CppModelHawkesExpLogLikelihoodSingle(2).compute_loss(path, END_TIME, 1.0, theta, True)
    path = [np.array([0.5, 1.0, 2.0]), np.array([1.5])]
    with pytest.raises(ValueError, match="C-contiguous"):
        CppModelHawkesExpLogLikelihoodSingle(2).compute_loss(
            path, END_TIME, 1.0, np.asfortranarray(theta), True)
