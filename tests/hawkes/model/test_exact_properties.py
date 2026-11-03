# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Model tests, layer 3: properties that must hold exactly on any path."""

import numpy as np
import pytest
from hawkes_numpy_reference import tiny_paths

from sparklen.hawkes.model import (
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)

BETA = 1.5
END_TIME = 5.0
MODELS = [ModelHawkesExpLeastSquares, ModelHawkesExpLogLikelihood]


def make_model(cls, data, decay=BETA, end_time=END_TIME):
    model = cls(decay=decay)
    model.set_data(data, end_time=end_time)
    return model


@pytest.fixture
def paths():
    return tiny_paths(n_paths=3, n_components=3, end_time=5.0, seed=1)


@pytest.fixture
def theta():
    rng = np.random.default_rng(2)
    return np.hstack([rng.uniform(0.2, 1.0, (3, 1)), rng.uniform(0.0, 0.3, (3, 3))])


def test_least_squares_changing_the_time_unit(paths, theta):
    """Check that changing the time unit by a factor c divides
    the least-squares loss by c^2."""
    c = 3.0
    model = make_model(ModelHawkesExpLeastSquares, paths, end_time=5.0)
    scaled = make_model(ModelHawkesExpLeastSquares,
                        [[c * events for events in path] for path in paths],
                        decay=BETA / c, end_time=c * 5.0)
    scaled_theta = np.hstack([theta[:, :1] / c, theta[:, 1:]])
    assert scaled.loss(scaled_theta) == pytest.approx(model.loss(theta) / c**2, rel=1e-12)


def test_log_likelihood_changing_the_time_unit(paths, theta):
    """Check that changing the time unit by a factor c shifts
    the log-likelihood by -N log c."""
    c = 3.0
    model = make_model(ModelHawkesExpLogLikelihood, paths, end_time=5.0)
    scaled = make_model(ModelHawkesExpLogLikelihood,
                        [[c * events for events in path] for path in paths],
                        decay=BETA / c, end_time=c * 5.0)
    scaled_theta = np.hstack([theta[:, :1] / c, theta[:, 1:]])
    n_events = sum(len(events) for path in paths for events in path)
    expected = model.loss(theta, neg=False) - n_events * np.log(c)
    assert scaled.loss(scaled_theta, neg=False) == pytest.approx(expected, rel=1e-12)


def test_lipschitz_constant_is_a_real_float(paths):
    """Check that the Lipschitz constant is a real float equal to the largest
    eigenvalue of the least-squares Hessian."""
    model = make_model(ModelHawkesExpLeastSquares, paths)
    lipschitz = model.lipschitz_const()
    assert isinstance(lipschitz, float)
    assert lipschitz == pytest.approx(np.max(np.linalg.eigvalsh(model.hessian())))
