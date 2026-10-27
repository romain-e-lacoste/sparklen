# Author: Romain E. Lacoste
# License: BSD-3-Clause

from tabulate import tabulate

from sparklen.optim.lr.base.learning_rate import LearningRateScheduler


class LipschitzLR(LearningRateScheduler):
    """
    Learning rate scheduler class.

    This class implements an adaptive step size adjustment procedure
    based on the Lipschitz constant of the gradient of the loss function.
    """
    def __init__(self):
        super().__init__()
        # if model._is_lipschitz == 0:
        #     raise NotImplementedError("The objective functional for this model do not have a Lipschitz continuous gradient")
        self._is_computed = False

    def step(self, search_point, loss_search_point, grad_search_point):
        """
        Determine the step size at a given iteration of the descent
        using the Lipschitz constant of the gradient of the loss function.

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
            The computed step size based on the Lipschitz constant.

        tentative_point : ndarray
            The tentative point after applying the computed step size.

        loss_tentative_point : float
            The loss evaluated at the tentative point.

        grad_tentative_point : ndarray
        The gradient evaluated at the tentative point.
        """
        #print("step", self._step_size)
        super().step(search_point, loss_search_point, grad_search_point)
        if not self._is_computed:
            lipschitz_constant = self._model.lipschitz_const()
            self._step_size = 1.0 / lipschitz_constant
            self._is_computed = True

        # Compute the tentative point with proximal operator
        tentative_point = search_point - self._step_size * grad_search_point
        self._prox.apply(tentative_point, self._step_size)

        # Compute the loss at the tentative point
        loss_tentative_point = self._model.loss(tentative_point)

        # Compute the gradient at the tentative point
        grad_tentative_point = self._model.grad(tentative_point)

        return self._step_size, tentative_point, loss_tentative_point, grad_tentative_point

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Learning rate scheduler", "Lipschitz constant"],
                 ["Current step size", self._step_size]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
