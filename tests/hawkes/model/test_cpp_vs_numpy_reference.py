# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Model tests, layer 2: the C++ core against an independent NumPy reference."""

import numpy as np
import pytest
from hawkes_numpy_reference import (
    finite_difference_grad,
    least_squares_loss,
    log_likelihood,
    tiny_paths,
)

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


def test_least_squares_loss_matches_reference(data, theta):
    """Check that the least-squares loss matches the NumPy reference."""
    model = make_model(ModelHawkesExpLeastSquares, data)
    expected = least_squares_loss(data, END_TIME, DECAY, theta)
    assert model.loss(theta) == pytest.approx(expected, rel=1e-9)


def test_least_squares_grad_matches_finite_differences(data, theta):
    """Check that the least-squares gradient matches finite differences of the loss."""
    model = make_model(ModelHawkesExpLeastSquares, data)
    expected = finite_difference_grad(model.loss, theta)
    np.testing.assert_allclose(model.grad(theta), expected, rtol=1e-6, atol=1e-8)


def test_log_likelihood_loss_matches_reference(data, theta):
    """Check that the log-likelihood matches the NumPy reference,
    and the loss its negation."""
    model = make_model(ModelHawkesExpLogLikelihood, data)
    expected = log_likelihood(data, END_TIME, DECAY, theta)
    assert model.loss(theta) == pytest.approx(-expected, rel=1e-9)
    assert model.loss(theta, neg=False) == pytest.approx(expected, rel=1e-9)


def test_log_likelihood_grad_matches_finite_differences(data, theta):
    """Check that the log-likelihood gradient matches finite differences of the loss."""
    model = make_model(ModelHawkesExpLogLikelihood, data)
    expected = finite_difference_grad(model.loss, theta)
    np.testing.assert_allclose(model.grad(theta), expected, rtol=1e-6, atol=1e-8)
