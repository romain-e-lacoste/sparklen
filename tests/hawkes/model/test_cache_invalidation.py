# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""The C++ model caches weights computed from the data, the end time and the
decay: changing any of them must rebuild it."""

import numpy as np
import pytest
from hawkes_numpy_reference import tiny_paths

from sparklen.hawkes.model import (
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)

END_TIME = 5.0
DECAY = 1.5


@pytest.fixture
def data():
    return tiny_paths(n_paths=3, n_components=2, end_time=END_TIME)


@pytest.fixture
def theta():
    return np.array([[0.4, 0.3, 0.1],
                     [0.2, 0.05, 0.25]])


def make_model(cls, data):
    model = cls(decay=DECAY)
    model.set_data(data, end_time=END_TIME)
    return model


@pytest.mark.parametrize("cls", [ModelHawkesExpLeastSquares, ModelHawkesExpLogLikelihood])
def test_changing_the_decay_updates_the_results(cls, data, theta):
    """Check that changing the decay rebuilds the C++ model: the loss and gradient
    match those of a model created with the new decay."""
    model = make_model(cls, data)
    model.loss(theta)  # fills the C++ cache with DECAY
    with pytest.warns(UserWarning, match="already been set"):
        model.decay = 2 * DECAY
    fresh = cls(decay=2 * DECAY)
    fresh.set_data(data, end_time=END_TIME)
    assert model.loss(theta) == fresh.loss(theta)
    np.testing.assert_array_equal(model.grad(theta), fresh.grad(theta))


@pytest.mark.parametrize("cls", [ModelHawkesExpLeastSquares, ModelHawkesExpLogLikelihood])
def test_changing_the_end_time_updates_the_results(cls, data, theta):
    """Check that setting the data with a new end time rebuilds the C++ model."""
    model = make_model(cls, data)
    model.loss(theta)
    with pytest.warns(UserWarning, match="already been set"):
        model.set_data(data, end_time=2 * END_TIME)
    fresh = cls(decay=DECAY)
    fresh.set_data(data, end_time=2 * END_TIME)
    assert model.loss(theta) == fresh.loss(theta)
