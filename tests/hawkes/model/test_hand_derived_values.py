# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Model tests, layer 1: comparison against hand-derived values."""

import numpy as np
import pytest

from sparklen.hawkes.model import (
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)

T = 4.0
BETA = 1.5
MU = 0.7
ALPHA = 0.4
T1, T2 = 1.0, 2.2
THETA = np.array([[MU, ALPHA]])
MODELS = [ModelHawkesExpLeastSquares, ModelHawkesExpLogLikelihood]


def make_model(cls, data, decay=BETA, end_time=T):
    model = cls(decay=decay)
    model.set_data(data, end_time=end_time)
    return model


def test_least_squares_empty_path():
    """Check that the least-squares loss of a path with no event is mu^2."""
    model = make_model(ModelHawkesExpLeastSquares, [[np.array([])]])
    assert model.loss(THETA) == pytest.approx(MU**2, rel=1e-14)


def test_log_likelihood_empty_path():
    """Check that the log-likelihood of a path with no event is -mu T."""
    model = make_model(ModelHawkesExpLogLikelihood, [[np.array([])]])
    assert model.loss(THETA, neg=False) == pytest.approx(-MU * T, rel=1e-14)


def test_least_squares_one_event():
    """Check the least-squares loss of a one-event path against its value by hand."""
    model = make_model(ModelHawkesExpLeastSquares, [[np.array([T1])]])
    E1 = np.exp(-BETA * (T - T1))
    square = MU**2 * T + 2 * MU * ALPHA * (1 - E1) + ALPHA**2 * BETA / 2 * (1 - E1**2)
    assert model.loss(THETA) == pytest.approx((square - 2 * MU) / T, rel=1e-14)


def test_least_squares_two_events():
    """Check the least-squares loss of a two-event path against its value by hand,
    which involves the pair term between the two events."""
    model = make_model(ModelHawkesExpLeastSquares, [[np.array([T1, T2])]])
    E1, E2 = np.exp(-BETA * (T - T1)), np.exp(-BETA * (T - T2))
    D = np.exp(-BETA * (T2 - T1))
    square = (MU**2 * T + 2 * MU * ALPHA * (2 - E1 - E2)
              + ALPHA**2 * BETA / 2 * ((1 - E1**2) + (1 - E2**2) + 2 * D * (1 - E2**2)))
    jumps = MU + (MU + ALPHA * BETA * D)
    assert model.loss(THETA) == pytest.approx((square - 2 * jumps) / T, rel=1e-14)


def test_log_likelihood_two_events():
    """Check the log-likelihood of a two-event path against its value by hand,
    and the loss as its negation."""
    model = make_model(ModelHawkesExpLogLikelihood, [[np.array([T1, T2])]])
    E1, E2 = np.exp(-BETA * (T - T1)), np.exp(-BETA * (T - T2))
    expected = (np.log(MU) + np.log(MU + ALPHA * BETA * np.exp(-BETA * (T2 - T1)))
                - MU * T - ALPHA * (2 - E1 - E2))
    assert model.loss(THETA, neg=False) == pytest.approx(expected, rel=1e-14)
    assert model.loss(THETA) == pytest.approx(-expected, rel=1e-14)


@pytest.mark.parametrize("cls", MODELS)
def test_poisson_estimate_is_the_event_rate(cls):
    """Check that, with alpha = 0, the gradient in mu vanishes at the rate N / T."""
    events = np.array([0.3, 1.1, 2.0, 3.7])
    model = make_model(cls, [[events]])
    grad = model.grad(np.array([[len(events) / T, 0.0]]))
    assert grad[0, 0] == pytest.approx(0.0, abs=1e-14)
