# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Calibration of the penalization constant."""

import numpy as np
import pytest

from sparklen.calibration import CalibrationCV, CalibrationEBIC
from sparklen.hawkes.simulation import SimuHawkesExp

DECAY = 2.0
END_TIME = 10.0


@pytest.fixture(scope="module")
def data():
    mu = np.array([0.5, 0.3])
    alpha = np.array([[0.3, 0.1], [0.0, 0.2]])
    simu = SimuHawkesExp(mu, alpha, DECAY, END_TIME, n_samples=10, random_state=1)
    return simu.simulate().timestamps


def make_calibration(cls):
    if cls is CalibrationCV:
        return CalibrationCV(cv=2, max_iter=30, verbose_bar=False, verbose=False)
    return CalibrationEBIC(max_iter=30, verbose_bar=False, verbose=False)


@pytest.mark.parametrize("cls", [CalibrationCV, CalibrationEBIC])
def test_calibrate_returns_self(cls, data):
    """Check that calibrate returns self and picks a constant from the grid."""
    calibration = make_calibration(cls)
    result = calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3,
                                   refinement=False)
    assert result is calibration
    assert calibration.best_kappa in (0.0, 0.1, 0.2)
    assert np.isfinite(calibration.best_score)


@pytest.mark.parametrize("refinement, expected", [(False, 0.1), (True, 0.13)])
@pytest.mark.parametrize("cls, method", [(CalibrationCV, "_cv_score"),
                                         (CalibrationEBIC, "_ebic_score")])
def test_calibrate_selects_the_best_constant(cls, method, refinement, expected,
                                             data, monkeypatch):
    """Check that calibrate picks the constant with the lowest score, on the coarse grid
    and after refinement."""
    calibration = make_calibration(cls)
    monkeypatch.setattr(calibration, method, lambda kappa, *args: (kappa - 0.13) ** 2)
    calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3, grid_step=0.1,
                          refinement=refinement, refined_grid_step=0.01)
    assert calibration.best_kappa == expected


@pytest.mark.parametrize("cls, method", [(CalibrationCV, "_cv_score"),
                                         (CalibrationEBIC, "_ebic_score")])
def test_errors_during_the_search_are_not_swallowed(cls, method, data, monkeypatch):
    """Check that an error raised while scoring reaches the caller."""
    calibration = make_calibration(cls)

    def fail(*args, **kwargs):
        raise RuntimeError("scoring failure")

    monkeypatch.setattr(calibration, method, fail)
    with pytest.raises(RuntimeError, match="scoring failure"):
        calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3)


@pytest.mark.parametrize("cls, method", [(CalibrationCV, "_cv_score"),
                                         (CalibrationEBIC, "_ebic_score")])
@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_no_finite_score_raises_a_clear_error(cls, method, bad_score, data,
                                              monkeypatch):
    """Check that a clear RuntimeError is raised when no score is finite."""
    calibration = make_calibration(cls)
    monkeypatch.setattr(calibration, method, lambda *args, **kwargs: bad_score)
    with pytest.raises(RuntimeError, match="not finite for any"):
        calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("cls, method", [(CalibrationCV, "_cv_score"),
                                         (CalibrationEBIC, "_ebic_score")])
def test_non_finite_scores_are_skipped_with_a_warning(cls, method, bad_score, data,
                                                      monkeypatch):
    """Check that a constant with a non-finite score is skipped with a warning
    and never selected, even with a score of -inf."""
    calibration = make_calibration(cls)

    def score(kappa, *args):
        return bad_score if kappa == 0.0 else (kappa - 0.13) ** 2

    monkeypatch.setattr(calibration, method, score)
    with pytest.warns(RuntimeWarning, match=r"kappa in \[0\.0\]"):
        calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3, grid_step=0.1,
                              refinement=False)
    assert calibration.best_kappa == 0.1


@pytest.mark.parametrize("cls, method", [(CalibrationCV, "_cv_score"),
                                         (CalibrationEBIC, "_ebic_score")])
def test_failed_calibration_leaves_no_result(cls, method, data, monkeypatch):
    """Check that a failed calibration does not keep the result of an earlier one."""
    calibration = make_calibration(cls)
    calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3, refinement=False)
    assert calibration.best_kappa is not None

    def fail(*args, **kwargs):
        raise RuntimeError("scoring failure")

    monkeypatch.setattr(calibration, method, fail)
    with pytest.raises(RuntimeError, match="scoring failure"):
        calibration.calibrate(DECAY, data, END_TIME, grid_max=0.3)
    assert calibration.best_kappa is None
    assert calibration.best_score is None


def test_star_import_exposes_both_calibrations():
    """Check that `from sparklen.calibration import *` exposes both classes."""
    namespace = {}
    exec("from sparklen.calibration import *", namespace)
    assert "CalibrationEBIC" in namespace
    assert "CalibrationCV" in namespace


def test_cross_validation_needs_enough_repetitions(data):
    """Check that more folds than repetitions raise a ValueError."""
    calibration = CalibrationCV(cv=20, verbose_bar=False, verbose=False)
    with pytest.raises(ValueError, match="at least 20 repetitions"):
        calibration.calibrate(DECAY, data, END_TIME)
