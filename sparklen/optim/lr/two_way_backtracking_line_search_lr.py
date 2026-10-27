# Author: Romain E. Lacoste
# License: BSD-3-Clause

import numpy as np
from numpy.linalg import norm
from tabulate import tabulate

from sparklen.optim.lr.base.learning_rate import LearningRateScheduler


class TwoWayBacktrackingLineSearchLR(LearningRateScheduler):
    """
    Learning rate scheduler class.

    This class implements a two-way backtracking line-search to dynamically
    adjust the step size at each iteration, ensuring stable and
    efficient convergence of the descent process.

    Parameters
    ----------
    tau : float, default=0.5
        A search control parameter that reduces the step size iteratively
        until the specified condition is met.
    """

    def __init__(self, tau=0.5):
        super().__init__()
        self._tau = tau

        self._step_size = 1.0

    def step(self, search_point, loss_search_point, grad_search_point):
        """
        Determine the step size at a given iteration of the descent
        using two-way backtracking line search.

        Parameters
        ----------
        search_point : ndarray
            Current point of the descent
        loss_search_point : float
            Loss evaluated at the current point.
        grad_search_point : ndarray
            Gradient evaluated at the current point.

        Returns
        -------
        step_size : float
            The computed step size based on line search.

        tentative_point : ndarray
            The tentative point after applying the computed step size.

        loss_tentative_point : float
            The loss evaluated at the tentative point.

        grad_tentative_point : ndarray
        The gradient evaluated at the tentative point.
        """
        super().step(search_point, loss_search_point, grad_search_point)

        sigma = self._step_size
        decreased = False  # Flag to track if step size was reduced

        while True:
            # Compute the tentative point with proximal operator
            tentative_point = search_point - sigma * grad_search_point
            self._prox.apply(tentative_point, sigma)

            # Compute the loss at the tentative point
            loss_tentative_point = self._model.loss(tentative_point)

            # Calculate the envelope
            envelope = loss_search_point + np.sum(grad_search_point * (tentative_point - search_point), axis=None) + 1. / (2 * sigma) * norm(tentative_point - search_point)**2

            # Check if the condition is satisfied using the precomputed envelope
            if loss_tentative_point <= envelope:
                if decreased:
                    break
                if sigma / self._tau > 1.0:
                    break # Accept step and exit
                sigma /= self._tau

            # Otherwise, reduce step size
            else:
                sigma *= self._tau # Reduce step size
                decreased = True

            # Break if step size is too small to avoid infinite loop
            if sigma < 1e-10:  # Example threshold
                print("Warning: Step size became too small.")
                break

        self._step_size = sigma

        # Compute the gradient at the tentative point
        grad_tentative_point = self._model.grad(tentative_point)

        return self._step_size, tentative_point, loss_tentative_point, grad_tentative_point

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Learning rate scheduler", "Two-Way Backtracking line-search"],
                 ["Search decrease parameter", self._tau],
                 ["Current step size", self._step_size]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
