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


@pytest.mark.parametrize("cls", MODELS)
def test_relabelling_the_components(cls, paths, theta):
    """Check that permuting the components leaves the loss unchanged
    and permutes the gradient accordingly."""
    # The permuted theta is Fortran-ordered, which the C++ core misread before 1.1.1.
    perm = np.array([2, 0, 1])
    permuted_theta = np.hstack([theta[perm, :1], theta[perm][:, 1 + perm]])
    model = make_model(cls, paths, end_time=5.0)
    permuted = make_model(cls, [[path[j] for j in perm] for path in paths], end_time=5.0)
    assert permuted.loss(permuted_theta) == pytest.approx(model.loss(theta), rel=1e-12)
    grad = model.grad(theta)
    expected = np.hstack([grad[perm, :1], grad[perm][:, 1 + perm]])
    np.testing.assert_allclose(permuted.grad(permuted_theta), expected, rtol=1e-12)


def test_lipschitz_constant_is_a_real_float(paths):
    """Check that the Lipschitz constant is a real float equal to the largest
    eigenvalue of the least-squares Hessian."""
    model = make_model(ModelHawkesExpLeastSquares, paths)
    lipschitz = model.lipschitz_const()
    assert isinstance(lipschitz, float)
    assert lipschitz == pytest.approx(np.max(np.linalg.eigvalsh(model.hessian())))


def least_squares_minimiser(model, d):
    # The least-squares loss is quadratic, row by row, with the same Hessian H
    # for every row: grad(theta)_i = H theta_i - b_i with b_i = -grad(0)_i, so
    # the minimiser solves H theta_i = b_i.
    b = -model.grad(np.zeros((d, d + 1)))
    return np.linalg.solve(model.hessian(), b.T).T


def test_least_squares_minimiser_solves_the_normal_equations(paths):
    """Check that the minimiser given by the normal equations has a zero gradient."""
    model = make_model(ModelHawkesExpLeastSquares, paths, end_time=5.0)
    minimiser = least_squares_minimiser(model, 3)
    np.testing.assert_allclose(model.grad(minimiser), 0.0, atol=1e-10)


@pytest.mark.parametrize("lr_scheduler", ["lipschitz", "backtracking", "fast-backtracking"])
@pytest.mark.parametrize("optimizer_name", ["gd", "agd"])
def test_optimisers_reach_the_least_squares_minimiser(paths, optimizer_name, lr_scheduler):
    """Check that GD and AGD, with each step-size rule,
    reach the exact least-squares minimiser."""
    from sparklen.optim.optimizer import AGD, GD
    from sparklen.prox import ProxZero

    model = make_model(ModelHawkesExpLeastSquares, paths, end_time=5.0)
    expected = least_squares_minimiser(model, 3)

    prox = ProxZero(positive=False)  # unconstrained, like the normal equations
    prox.set_pen_const(0.0)
    prox.set_application_range(0, 4)
    cls = GD if optimizer_name == "gd" else AGD
    optimizer = cls(lr_scheduler, max_iter=5000, tol=1e-14,
                    verbose_bar=False, verbose=False)
    optimizer.set_model(model)
    optimizer.set_prox(prox)
    optimizer.optimize(np.full((3, 4), 0.2))

    # The stopping rule watches the loss, so the loss gap is tight while the
    # parameters, on this ill-conditioned problem, are only close.
    gap = model.loss(optimizer.minimizer) - model.loss(expected)
    assert 0 <= gap <= 1e-9 * abs(model.loss(expected))
    np.testing.assert_allclose(optimizer.minimizer, expected, atol=1e-3)
