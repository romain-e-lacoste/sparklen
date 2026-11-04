# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Behaviour of the optimizers on a simple quadratic problem."""

import numpy as np
import pytest

from sparklen.optim.optimizer import AGD, GD
from sparklen.optim.optimizer.base.optimizer import Optimizer
from sparklen.prox import ProxZero

TARGET = np.array([[1.0, 0.5, 0.2],
                   [0.3, 0.0, 0.7]])
X0 = np.zeros((2, 3))


class Quadratic:
    """f(x) = scale * ||x - TARGET||^2 / 2 + offset.

    With fault="error" or "nan", the loss raises or returns NaN away from the
    starting point X0, so from the first step on.
    """

    def __init__(self, fault=None, scale=1.0, offset=0.0):
        self.fault = fault
        self.scale = scale
        self.offset = offset
        self.evaluated_at_start = False  # set once the loss is evaluated at X0

    def check_set_state(self):
        pass

    def loss(self, x):
        if np.array_equal(x, X0):
            self.evaluated_at_start = True
        elif self.fault is not None:
            if self.fault == "error":
                raise RuntimeError("model failure")
            return np.nan
        return 0.5 * self.scale * np.sum((x - TARGET) ** 2) + self.offset

    def grad(self, x):
        return self.scale * (x - TARGET)

    def lipschitz_const(self):
        return self.scale


def make_optimizer(cls, model, lr_scheduler="backtracking", verbose_bar=False,
                   max_iter=200, tol=1e-10):
    prox = ProxZero()
    prox.set_pen_const(0.0)
    prox.set_application_range(0, 3)
    optimizer = cls(lr_scheduler, max_iter=max_iter, tol=tol,
                    verbose_bar=verbose_bar, verbose=False, record_every=1)
    optimizer.set_model(model)
    optimizer.set_prox(prox)
    return optimizer


@pytest.mark.parametrize("cls", [GD, AGD])
def test_optimize_returns_self_and_converges(cls):
    """Check that optimize returns self and converges to the minimiser."""
    optimizer = make_optimizer(cls, Quadratic())
    assert optimizer.optimize(np.zeros((2, 3))) is optimizer
    np.testing.assert_allclose(optimizer.minimizer, TARGET, atol=1e-4)


@pytest.mark.parametrize("cls", [GD, AGD])
def test_minimizer_is_the_newest_iterate(cls):
    """Check that the returned minimizer is the last iterate, not the previous one."""
    optimizer = make_optimizer(cls, Quadratic(scale=0.5, offset=1.0), tol=1e-3)
    optimizer.optimize(np.zeros((2, 3)))
    np.testing.assert_array_equal(optimizer.minimizer, optimizer.history["x"][-1])


@pytest.mark.parametrize("cls", [GD, AGD])
@pytest.mark.parametrize("verbose_bar", [False, True])
def test_errors_are_not_swallowed(cls, verbose_bar, capsys):
    """Check that an error raised at the first step reaches the caller and leaves
    no result, with or without the progress bar."""
    optimizer = make_optimizer(cls, Quadratic(), verbose_bar=verbose_bar)
    optimizer.optimize(X0.copy())  # a successful run first
    capsys.readouterr()
    failing = Quadratic(fault="error")
    optimizer.set_model(failing)
    with pytest.raises(RuntimeError, match="model failure"):
        optimizer.optimize(X0.copy())
    assert failing.evaluated_at_start  # the failure came after the start, in a step
    with pytest.raises(ValueError, match="not been completed"):
        optimizer.minimizer  # no stale result from the first run
    captured = capsys.readouterr()
    assert "Optimization" not in captured.out + captured.err  # no completion message


@pytest.mark.parametrize("cls", [GD, AGD])
def test_non_finite_loss_raises(cls):
    """Check that a NaN loss at the first step raises FloatingPointError."""
    optimizer = make_optimizer(cls, Quadratic(fault="nan"))
    with pytest.raises(FloatingPointError, match="not finite"):
        optimizer.optimize(X0.copy())


@pytest.mark.parametrize("cls", [GD, AGD])
def test_zero_loss_does_not_divide_by_zero(cls):
    """Check that starting at the minimiser, where the loss is 0, does not divide
    by zero and stops after one step."""
    optimizer = make_optimizer(cls, Quadratic())
    optimizer.optimize(TARGET.copy())
    np.testing.assert_allclose(optimizer.minimizer, TARGET)
    assert len(optimizer.history["x"]) == 1  # the loss did not move: converged at once


@pytest.mark.parametrize("loss_new, loss_old, expected", [
    (2.0, 1.0, 1.0),
    (-0.5, -1.0, 0.5),    # negative losses are common (least squares, -log L)
    (0.5, -1.0, 1.5),     # the loss changes sign
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),      # the loss did not move: converged
    (1e-3, 0.0, np.inf),  # the loss moved away from 0: not converged
])
def test_relative_change_handles_negative_and_zero_losses(loss_new, loss_old, expected):
    """Check the relative change of the loss, including negative and zero losses."""
    assert Optimizer._relative_change(loss_new, loss_old) == expected


@pytest.mark.parametrize("loss_new, loss_old", [(np.nan, 1.0), (np.inf, np.inf)])
def test_non_finite_relative_change_never_converges(loss_new, loss_old):
    """Check that a non-finite relative change never passes the convergence test."""
    assert not Optimizer._relative_change(loss_new, loss_old) < 1e-10


@pytest.mark.parametrize("loss", [np.nan, np.inf, -np.inf])
def test_check_finite_rejects_nan_and_infinities(loss):
    """Check that a NaN or infinite loss raises FloatingPointError."""
    with pytest.raises(FloatingPointError, match="iteration 3"):
        Optimizer._check_finite(loss, iteration=3)


def test_lipschitz_step_is_recomputed_for_a_new_model():
    """Check that the Lipschitz step size 1/L is recomputed for a new model."""
    optimizer = make_optimizer(GD, Quadratic(scale=1.0), lr_scheduler="lipschitz",
                               max_iter=5)
    optimizer.optimize(np.zeros((2, 3)))
    assert optimizer.history["lr"][-1] == pytest.approx(1.0)

    optimizer.set_model(Quadratic(scale=4.0))
    optimizer.optimize(np.zeros((2, 3)))
    assert optimizer.history["lr"][-1] == pytest.approx(0.25)


@pytest.mark.parametrize("cls", [GD, AGD])
def test_print_info_runs(cls, capsys):
    """Check that print_info runs and names the step-size rule."""
    optimizer = make_optimizer(cls, Quadratic())
    optimizer.print_info()
    assert "BacktrackingLineSearchLR" in capsys.readouterr().out


@pytest.mark.parametrize("cls", [GD, AGD])
def test_set_prox_checks_the_prox_state(cls):
    """Check that set_prox rejects an operator without a penalization constant."""
    optimizer = cls("backtracking", max_iter=10, tol=1e-5,
                    verbose_bar=False, verbose=False)
    with pytest.raises(AttributeError, match="penalization constant"):
        optimizer.set_prox(ProxZero())
