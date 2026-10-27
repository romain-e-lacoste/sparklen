# Author: Romain E. Lacoste
# License: BSD-3-Clause

import numpy as np
from numpy.linalg import norm
from tabulate import tabulate

from sparklen.optim.lr.base.learning_rate import LearningRateScheduler


class BacktrackingLineSearchLR(LearningRateScheduler):
    """
    Learning rate scheduler class.

    This class implements a backtracking line-search to dynamically
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

    def step(self, search_point, loss_search_point, grad_search_point):
        """
        Determine the step size at a given iteration of the descent
        using backtracking line search.

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

        self._step_size = 1.0

        while True:
            # Compute the tentative point with proximal operator
            #print("step", self._step_size)

            tentative_point = search_point - self._step_size * grad_search_point
            self._prox.apply(tentative_point, self._step_size)

            # Compute the loss at the tentative point
            loss_tentative_point = self._model.loss(tentative_point)

            # Calculate the envelope
            envelope = loss_search_point + np.sum(grad_search_point * (tentative_point - search_point), axis=None) + 1. / (2 * self._step_size) * norm(tentative_point - search_point)**2

            # Check if the condition is satisfied using the precomputed envelope
            if loss_tentative_point <= envelope:
                break # Armijo condition satisfied, exit loop
            self._step_size *= self._tau # Reduce step size

            # Break if step size is too small to avoid infinite loop
            if self._step_size < 1e-10:  # Example threshold
                print("Warning: Step size became too small.")
                break

        # Compute the gradient at the tentative point
        grad_tentative_point = self._model.grad(tentative_point)

        return self._step_size, tentative_point, loss_tentative_point, grad_tentative_point

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Learning rate scheduler", "Backtracking line-search"],
                 ["Search decrease parameter", self._tau],
                 ["Current step size", self._step_size]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
