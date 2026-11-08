# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Proximal operators against their definition and their closed forms.

For a step size s and a penalty constant kappa, the proximal operator of a
penalty Omega maps v to

    prox(v) = argmin_u  (u - v)^2 / 2 + s kappa Omega(u),

entry by entry on the columns start:end, the other columns being left
unchanged. With positive=True, u is also constrained to be non-negative.
The penalties follow scikit-learn's ElasticNet convention (the Lasso is
l1_ratio = 1, the Ridge l1_ratio = 0):

    Lasso        Omega(u) = |u|
    Ridge        Omega(u) = u^2 / 2
    Elastic-Net  Omega(u) = r |u| + (1 - r) u^2 / 2,   r = l1_ratio

which give, with t = s kappa and S(v, a) = sign(v) max(|v| - a, 0):

    Lasso        S(v, t)
    Ridge        v / (1 + t)
    Elastic-Net  S(v, t r) / (1 + t (1 - r))

With positive=True, the minimiser over u >= 0 is max(prox(v), 0); since each
operator is non-decreasing and maps 0 to 0, this is the same formula applied
to max(v, 0).
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from sparklen.prox import ProxElasticNet, ProxL1, ProxL2, ProxZero

STEP = 0.5
KAPPA = 0.8
RATIO = 0.3
T = STEP * KAPPA  # 0.4
# Values on both sides of 0 and of the thresholds t and t r
VALUES = np.array([-2.0, -0.5, -0.4, -0.3, -0.1, 0.0, 0.05, 0.12, 0.3, 0.4, 0.45, 1.0, 3.0])


def soft_threshold(v, a):
    return np.sign(v) * np.maximum(np.abs(v) - a, 0.0)


def closed_form(name, v, positive):
    if positive:
        v = np.maximum(v, 0.0)
    if name == "zero":
        return v
    if name == "lasso":
        return soft_threshold(v, T)
    if name == "ridge":
        return v / (1 + T)
    return soft_threshold(v, T * RATIO) / (1 + T * (1 - RATIO))


def penalty(name, u):
    if name == "zero":
        return 0.0
    if name == "lasso":
        return abs(u)
    if name == "ridge":
        return u**2 / 2
    return RATIO * abs(u) + (1 - RATIO) * u**2 / 2


def make_prox(name, positive):
    if name == "zero":
        prox = ProxZero(positive=positive)
    elif name == "lasso":
        prox = ProxL1(positive=positive)
    elif name == "ridge":
        prox = ProxL2(positive=positive)
    else:
        prox = ProxElasticNet(positive=positive, l1_ratio=RATIO)
    prox.set_pen_const(KAPPA)
    prox.set_application_range(0, len(VALUES))
    return prox


def apply(prox, values, step=STEP):
    x = np.array([values], dtype=np.float64)  # one row, C-contiguous
    prox.apply(x, step)
    return x[0]


NAMES = ["zero", "lasso", "ridge", "elasticnet"]


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_closed_form(name, positive):
    """Check that each proximal operator matches its closed form."""
    result = apply(make_prox(name, positive), VALUES)
    expected = closed_form(name, VALUES, positive)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_definition_as_a_minimiser(name, positive):
    """Check that each proximal operator returns the minimiser of its defining
    objective, found numerically."""
    result = apply(make_prox(name, positive), VALUES)
    for v, u in zip(VALUES, result):
        def objective(w, v=v):
            return (w - v) ** 2 / 2 + T * penalty(name, w)
        lower = 0.0 if positive else -5.0
        best = minimize_scalar(objective, bounds=(lower, 5.0), method="bounded",
                               options={"xatol": 1e-12})
        # The bounded method stops within sqrt(eps)|x| + xatol/3 < 1e-7 of the minimiser
        assert u == pytest.approx(best.x, abs=1e-7)
        assert objective(u) <= best.fun + 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_only_the_application_range_changes(name):
    """Check that only the columns in the application range are modified,
    on both sides of the range."""
    prox = make_prox(name, positive=False)
    prox.set_application_range(1, 3)
    x = np.array([[-1.0, -1.0, 0.2, 2.0, -3.0], [3.0, 0.1, -0.3, 1.5, 0.7]])
    expected = x.copy()
    expected[:, 1:3] = closed_form(name, x[:, 1:3], positive=False)
    prox.apply(x, STEP)
    np.testing.assert_allclose(x, expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("name", NAMES)
def test_zero_penalty_is_the_identity_or_the_projection(name):
    """Check that a zero penalty gives the identity,
    or the projection onto u >= 0 when positive."""
    for positive in (False, True):
        prox = make_prox(name, positive)
        prox.set_pen_const(0.0)
        expected = np.maximum(VALUES, 0.0) if positive else VALUES
        np.testing.assert_array_equal(apply(prox, VALUES), expected)


@pytest.mark.parametrize("positive", [False, True])
def test_elastic_net_extremes_are_lasso_and_ridge(positive):
    """Check that the elastic net with l1_ratio 1 or 0 is the lasso or the ridge."""
    lasso, ridge = make_prox("lasso", positive), make_prox("ridge", positive)
    for ratio, other in ((1.0, lasso), (0.0, ridge)):
        elastic = ProxElasticNet(positive=positive, l1_ratio=ratio)
        elastic.set_pen_const(KAPPA)
        elastic.set_application_range(0, len(VALUES))
        np.testing.assert_allclose(apply(elastic, VALUES), apply(other, VALUES),
                                   rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("name", NAMES)
def test_step_size_and_constant_enter_as_a_product(name):
    """Check that only the product of the step size and the penalization
    constant matters."""
    prox = make_prox(name, positive=False)
    doubled = make_prox(name, positive=False)
    doubled.set_pen_const(2 * KAPPA)
    np.testing.assert_allclose(apply(doubled, VALUES, step=STEP / 2),
                               apply(prox, VALUES), rtol=1e-14, atol=1e-15)
