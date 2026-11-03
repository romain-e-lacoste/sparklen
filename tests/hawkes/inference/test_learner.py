# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""LearnerHawkesExp: fitting, and the checks of its options."""

import numpy as np
import pytest

from sparklen.hawkes.inference import LearnerHawkesExp
from sparklen.hawkes.simulation import SimuHawkesExp

DECAY = 2.0
END_TIME = 10.0


@pytest.fixture(scope="module")
def data():
    mu = np.array([0.5, 0.3])
    alpha = np.array([[0.3, 0.1], [0.0, 0.2]])
    simu = SimuHawkesExp(mu, alpha, DECAY, END_TIME, n_samples=10, random_state=1)
    return simu.simulate().timestamps


def test_learner_fit_returns_self(data):
    """Check that LearnerHawkesExp.fit returns the learner itself,
    with a finite estimate."""
    learner = LearnerHawkesExp(decay=DECAY, penalty="none", max_iter=30,
                               verbose_bar=False, verbose=False)
    assert learner.fit(data, END_TIME) is learner
    assert learner.estimated_params.shape == (2, 3)
    assert np.all(np.isfinite(learner.estimated_params))


@pytest.mark.filterwarnings("ignore:.*has already been set:UserWarning")
def test_failed_fit_leaves_no_estimate(data, monkeypatch):
    """Check that a failed fit does not keep the estimate of an earlier one."""
    learner = LearnerHawkesExp(decay=DECAY, penalty="none", max_iter=30,
                               verbose_bar=False, verbose=False)
    learner.fit(data, END_TIME)

    def fail(*args, **kwargs):
        raise RuntimeError("optimization failure")

    monkeypatch.setattr(learner._optimizer, "optimize", fail)
    with pytest.raises(RuntimeError, match="optimization failure"):
        learner.fit(data, END_TIME)
    with pytest.raises(ValueError, match="call fit"):
        learner.estimated_params
    with pytest.raises(ValueError, match="call fit"):
        learner.score(data, END_TIME)
