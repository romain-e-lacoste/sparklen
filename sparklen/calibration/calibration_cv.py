# Author: Romain E. Lacoste
# License: BSD-3-Clause

import time

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from sparklen.calibration.base.calibration import Calibration
from sparklen.hawkes.model import ModelHawkesExpLogLikelihood


class CalibrationCV(Calibration):
    """
    Calibration class for Hawkes process with exponential kernel.

    This class implements a Cross-Validation-based methodology for tuning
    the regularization constant in the following optimization problem:

    .. math::
        \\hat{\\theta}_n(\\kappa) \\in \\arg\\min_{
        \\theta \in \\mathbb{R}^{d \\times d+1}}
        \\left\\{F_{T, n}(\\theta) + \\kappa \\Omega(\\theta) \\right\}

    where

    * :math:`\\kappa` is the regularization constant

    The strategy involves exploring a grid :math:`\\Delta` of :math:`\\kappa`
    values and selecting the one that minimizes the criterion of interest.
    In the case of Cross-Validation, this entails partitioning the dataset
    into folds and aiming to minimize the cross-validated risk

    Parameters
    ----------
    cv : int, default=5
        The number of cross-validation splits.

    loss : str, {'least-squares', 'log-likelihood'}, default='least-squares'
        Specifies the loss function to be used. The available options are:

            - 'least-squares': The least-squares loss function.
            - 'log-likelihood': The log-likelihood loss function.

    penalty : str, {'lasso', 'ridge', 'elasticnet'}, default='lasso'
        Specifies the type of penalty to be applied. The possible options are:

        - 'lasso': Lasso regularization (:math:`\ell_1`-penalty)
        - 'ridge': Ridge (or Tikhonov) regularization (:math:`\ell_2`-penalty)
        - 'elasticnet': Elastic-Net regularization (:math:`\ell_1 + \ell_2`-penalty)

    optimizer : str, {'gd', 'agd'}, default='agd'
        Specifies the optimization algorithm to use. The possible options are:

        - 'gd' : Gradient Descent (GD)
        - 'agd' : Accelerated Gradient Descent (AGD)

    lr_scheduler : str, {'lipschitz', 'backtracking'}, default='backtracking'
        Specifies the learning rate scheduler. The available options are:

        - 'lipschitz' : Lipschitz-based step size, usable only if `loss='least-squares'`
        - 'backtracking' : Backtracking line-search-based step size.
    max_iter : int, default=100
        The maximum number of iterations allowed during the optimization process.

    tol : float, default=1e-5
        The tolerance for the precision achieved during the optimization process.
        The optimization will stop when the convergence criterion falls below this value.
        If the tolerance is not reached, the optimizer will perform a maximum of `max_iter` iterations.

    penalty_mu : bool, default=False
        Determines whether the penalty is applied to `mu` (the first column of the parameter).
        If `penalty_mu=False`, the regularization is applied only to the interaction matrix.

    verbose_bar : bool, default=True
        Determines whether a progress bar is displayed during the searching process,
        along with information such as the actual `kappa` value and score.
        If `verbose_bar=False`, no information is displayed.

    verbose : bool, default=True
        Controls whether recorded information during the searching process is printed.
        If `verbose=False`, no information is printed.

    Attributes
    ----------
    best_kappa : float
        The best penalty constant found after calibration.

    best_score : float
        The best score achieved, namely the best cross-validated risk,
        associated with the `best_kappa` value.

    Notes
    ----------
    This class handle the calibration of univariate Hawkes processes, as these
    are naturally included as a special case of the multivariate model.
    """

    def __init__(self, cv=5, loss="least-squares", penalty="lasso",
                 optimizer="agd", lr_scheduler="backtracking",
                 max_iter=100, tol=1e-5, penalty_mu=False,
                 verbose_bar=True, verbose=True):

        # Call the initializer of the base class ModelHawkes
        super().__init__(loss, penalty, optimizer, lr_scheduler, max_iter, tol, penalty_mu, verbose_bar, verbose)

        self._cv = cv

    def calibrate(self, decay, data, end_time=None, grid_max=2.0, grid_step=0.1, refinement=True, refined_grid_step=0.01):
        """
        Calibrate the regularization constant given training data.

        Parameters
        ----------
        decay : float
            The decay hyperparameter of the exponential kernel of the process.
            This scalar dictates how quick the influences vanish over time.

        data : list of list of ndarray
            Repeated paths of a Hawkes process. The outer list has length `n`,
            representing the number of repetitions. Each inner list has length `d`,
            corresponding to the number of components (dimensions) of the Hawkes process.

            Specifically, `data[i][j]` is a one-dimensional `ndarray` containing
            the event times of the `j`-th component in the `i`-th realization.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.

        grid_max : float, default=2.0
            Upper bound of the searching grid.

        grid_step : float, default=0.1
            Step size of the searching grid.

        refinement : bool, default=True
            Determines whether another finer grid search is performed.
            If `refinement=True`, a finer grid search with the step
            `refined_grid_step` is performed around the best constant
            found by the previous, coarser grid search.

        refined_grid_step : float, default=0.01
            Step size of the finer searching grid.
            This parameter is only considered if `refinement='True'`

        Returns
        -------
        self : object
            The instance of the calibrated object.
        """
        self._model = self._losses[self._str_loss](decay)
        self._model.set_data(data, end_time)

        self._model_likelihood = ModelHawkesExpLogLikelihood(decay)
        self._model_likelihood.set_data(data, end_time)

        if self._penalty_mu:
            self._prox.set_application_range(0, self._model.n_components()+1)
        else:
            self._prox.set_application_range(1, self._model.n_components()+1)

        self._prox.set_pen_const(pen_const=0.0)

        self._optimizer.set_model(self._model)
        self._optimizer.set_prox(self._prox)

        coarse_grid = np.arange(0, grid_max, grid_step)
        best_kappa, best_score = self._search_grid(decay, data, end_time, coarse_grid)

        if refinement:
            # Perform a refined search around the best coarse parameter with refined_grid_step
            refined_grid_min = max(0, best_kappa - grid_step + refined_grid_step)  # Ensure lower bound is not negative
            refined_grid_max = min(grid_max, best_kappa + grid_step)  # Prevent exceeding upper bound
            refine_grid = np.arange(round(refined_grid_min, 2), round(refined_grid_max, 2), refined_grid_step)
            best_kappa, best_score = self._search_grid(decay, data, end_time, refine_grid, title="Refined Searching")

        self._best_kappa, self._best_score = best_kappa, best_score

    def _search_grid(self, decay, data, end_time, grid, title="Searching"):
        """
        Internal method to perform a grid search and return the best parameter and its BIC score.
        """

        start_time = time.time()  # Start the timer

        best_score = float('inf')
        best_kappa = None

        pbar = None
        if self._verbose_bar:
            # Setup the progress bar
            pbar = tqdm(total=len(grid), desc=title, unit="it")

        try:
            for kappa in grid:
                score = self._cv_score(kappa, decay, data, end_time)

                # Update progress bar with current kappa and score
                if self._verbose_bar:
                    pbar.update(1)
                    if self._verbose:
                        pbar.set_postfix({"kappa": kappa, "score": score})

                if score < best_score:
                    best_score = score
                    best_kappa = round(kappa, 2)

        except Exception as e:
            if pbar:
                pbar.write(f"\nCV search interrupted: {e}")

        finally:
            end_time = time.time()  # End the timer
            elapsed_time = end_time - start_time

            if pbar:
                pbar.write(f"\nBest kappa found: {best_kappa} with score: {best_score}")
                pbar.write(f"\nTime elapsed: {elapsed_time:.2f} seconds.")

                pbar.close()

        return best_kappa, best_score

    def _cv_score(self, kappa, decay, data, end_time):

        kf = KFold(n_splits=self._cv)
        scores = []
        for train_index, val_index in kf.split(data):
            data_train = [data[i] for i in train_index]
            data_val = [data[i] for i in val_index]

            model_train = self._losses[self._str_loss](decay)
            model_train.set_data(data_train, end_time)

            self._prox.set_pen_const(pen_const=kappa)

            theta_init = np.ones((model_train.n_components(), model_train.n_components()+1))*0.2
            self._optimizer.set_model(model_train)
            self._optimizer.set_prox(self._prox)
            self._optimizer.optimize(theta_init)
            theta_hat = self._optimizer.minimizer

            model_val = self._losses[self._str_loss](decay)
            model_val.set_data(data_val, end_time)
            score = model_val.loss(theta_hat)
            scores.append(score)
        avg_score = np.mean(scores)
        return avg_score

    def print_info(self):
        pass
