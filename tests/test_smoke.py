# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Smoke tests: the compiled extensions load and one small end-to-end run works.

These tests are deliberately small and fast. They check that a freshly built
wheel is usable, not that the estimators are statistically correct.
"""

import numpy as np
import pytest

import sparklen


def test_version_is_set():
    assert isinstance(sparklen.__version__, str)
    assert sparklen.__version__ != ""


def test_public_api_imports():
    from sparklen.calibration import CalibrationCV, CalibrationEBIC
    from sparklen.hawkes.classification import (
        ERMCLassifier,
        ERMLRCLassifier,
        make_classification,
    )
    from sparklen.hawkes.inference import LearnerHawkesExp
    from sparklen.hawkes.model import (
        ModelHawkesExpClassification,
        ModelHawkesExpLeastSquares,
        ModelHawkesExpLogLikelihood,
    )
    from sparklen.hawkes.simulation import SimuHawkesExp
    from sparklen.optim.lr import (
        BacktrackingLineSearchLR,
        LipschitzLR,
        TwoWayBacktrackingLineSearchLR,
    )
    from sparklen.optim.optimizer import AGD, GD
    from sparklen.prox import ProxElasticNet, ProxL1, ProxL2, ProxZero


@pytest.fixture(scope="module")
def small_sample():
    from sparklen.hawkes.simulation import SimuHawkesExp

    mu = np.array([0.5, 0.3])
    alpha = np.array([[0.3, 0.1], [0.0, 0.2]])
    simu = SimuHawkesExp(mu, alpha, beta=2.0, end_time=10.0, n_samples=20,
                         random_state=0)
    simu.simulate()
    return simu.timestamps


def test_simulation_shape(small_sample):
    assert len(small_sample) == 20
    for path in small_sample:
        assert len(path) == 2
        for events in path:
            assert events.dtype == np.float64
            assert np.all(np.diff(events) >= 0)
            assert np.all((events >= 0) & (events < 10.0))


def test_least_squares_loss_and_grad_are_finite(small_sample):
    from sparklen.hawkes.model import ModelHawkesExpLeastSquares

    model = ModelHawkesExpLeastSquares(decay=2.0)
    model.set_data(small_sample, end_time=10.0)
    theta = np.full((2, 3), 0.2)
    assert np.isfinite(model.loss(theta))
    grad = model.grad(theta)
    assert grad.shape == (2, 3)
    assert np.all(np.isfinite(grad))


def test_learner_fit_runs(small_sample):
    from sparklen.hawkes.inference import LearnerHawkesExp

    learner = LearnerHawkesExp(decay=2.0, loss="least-squares", penalty="none",
                               optimizer="agd", lr_scheduler="backtracking",
                               max_iter=50, verbose_bar=False, verbose=False)
    learner.fit(small_sample, end_time=10.0)
    theta = learner.estimated_params
    assert theta.shape == (2, 3)
    assert np.all(np.isfinite(theta))
