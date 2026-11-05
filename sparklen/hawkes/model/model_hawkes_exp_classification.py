# Author: Romain E. Lacoste
# License: BSD-3-Clause

import numpy as np
from scipy.special import logsumexp
from tabulate import tabulate

from sparklen.hawkes.model.base.model_hawkes_classification import (
    ModelHawkesClassification,
)
from sparklen.hawkes.model.build.hawkes_model import (
    ModelHawkesExpLogLikelihoodSingle as CppModelHawkesExpLogLikelihoodSingle,
)


class ModelHawkesExpClassification(ModelHawkesClassification):
    """
    Model  :math:`K`-classification class for Hawkes process with
    exponential kernel and given decay.

    This class handles model-related calculations, including loss evaluation,
    gradient computation, and Hessian matrix calculation

    The observed data consist of labeled repeated realizations of a Hawkes process.

    .. math::
        \\left\\{\\left(\\left\\{t_{j,\\ell}^{(i)}\\right\\}_{\\ell \\in [N_j(T)]},
        Y^{(i)}\\right), \ j \\in [d], \ i \\in [n] \\right\}

    where

    * :math:`K \\geq 2` is the number of classes
    * :math:`n \\geq 1` is the number of repetitions
    * :math:`d \\geq 1` is the number of components
    * :math:`T > 0` is the time horizon
    * :math:`t_{j,\\ell}^{(i)}` is the :math:`\\ell`-th event time of component :math:`j` in the :math:`i`-th repetition
    * :math:`Y^{(i)} \\in [K]` is the label of the :math:`i`-th feature

    The feature consists of time events of a Hawkes process whose dynamics
    of occurrence is governed by its label.
    The intensity function of the `j`-th process depends on :math:`Y` and is given by:

    .. math::
        \\mu_{Y,j} + \\sum_{j'=1}^d \\alpha_{Y,j,j'} \\int_0^t
        \\beta e^{-\\beta(t-s)} \ \\textrm{d} N_{j'}(s)

    where

    * :math:`(\\mu_{k,j})_{j \\in [d]}` is the vector of exogenous intensities of class `k`
    * :math:`(\\alpha_{k, j, j'})_{j, j' \\in [d]}` is the matrix of interactions of class `k`
    * :math:`\\beta` is the fixed and given decay of the exponential kernel

    """

    def __init__(self, decay=None, weights=None):
        # Call the initializer of the base class ModelHawkesClassification
        super().__init__(decay, weights)

    def set_data(self, X, y, end_time=None):
        """
        Initialize the model with data and a specified time horizon.

        Parameters
        ----------
        X : list of list of ndarray
            Repeated paths of a Hawkes process. The outer list has length `n`,
            representing the number of repetitions. Each inner list has length `d`,
            corresponding to the number of components (dimensions) of the Hawkes process.

            Specifically, `data[i][j]` is a one-dimensional `ndarray` containing
            the event times of the `j`-th component in the `i`-th realization.

        Y : ndarray of shape (n,)
            Associated class labels for each realization.

        end_time : float, default=None
            The end time of the observation period. The time horizon defines
            the interval `[0, T]` over which the Hawkes process is observed.

            - If `end_time` is provided, it is used as the upper bound of the observation window.
            - If `end_time=None`, it is automatically set to the largest observed event time
              across all components and repetitions.
        """
        # Call the base class logic
        super().set_data(X, y, end_time)

    @staticmethod
    def _class_log_likelihoods(jump_times, end_time, bold_theta, decay, with_grad=False):
        """
        Log-likelihood F_k of one path under each class parameter, and
        optionally its gradient with respect to each class parameter.
        """
        K = len(bold_theta)
        M = len(bold_theta[0])
        F = np.empty(K, dtype=np.float64)
        grad_F = np.empty(bold_theta.shape, dtype=np.float64) if with_grad else None
        for k in range(K):
            # One C++ model per class: its cached weights are reused by the
            # gradient call below.
            model = CppModelHawkesExpLogLikelihoodSingle(M)
            F[k] = model.compute_loss(jump_times, end_time, decay, bold_theta[k], False)
            if with_grad:
                grad_F[k] = model.compute_grad(jump_times, end_time, decay, bold_theta[k], False)
        return F, grad_F

    @staticmethod
    def _posterior(F, p):
        """
        Posterior class probabilities from the log-likelihoods F and the class
        weights p, computed in log space (logsumexp) so that large
        log-likelihoods neither overflow nor underflow.
        """
        with np.errstate(divide="ignore"):
            log_weighted = np.log(p) + F
        return np.exp(log_weighted - logsumexp(log_weighted))

    @staticmethod
    def _get_pi(jump_times, end_time, bold_theta, decay, p):
        """
        Posterior class probabilities of one path,
        pi_k = p_k exp(F_k) / sum_m p_m exp(F_m), where F_k is the log-likelihood
        of the path under the parameter of class k and p the class weights.
        Used by the loss and by predict_proba.
        """
        F, _ = ModelHawkesExpClassification._class_log_likelihoods(jump_times, end_time, bold_theta, decay)
        return ModelHawkesExpClassification._posterior(F, p)

    def loss(self, bold_theta):
        """
        Compute the value of the classification loss evaluated
        at the given input parameter.

        Parameters
        ----------
        theta : ndarray of shape (K, d, d+1)
            Parameter of the model. For each class `k`, the first column
            of `bold_theta[k]` corresponds to the exogenous intensity parameter.
            The `d` next columns correspond to the interaction matrix.

        Returns :
        -------
        float
            The value of the classification loss.
        """
        # Call the base class logic
        bold_theta = super().loss(bold_theta)

        result = 0.0
        for rep in range(self._n_repetitions):
            pi = self._get_pi(self._X[rep], self._end_time, bold_theta, self._decay, self._weights)
            # Squared distance between pi and the one-hot encoding of the label
            result += np.sum(pi**2) - 2*pi[self._y[rep]] + 1
        return (4*result)/self._n_repetitions

    def grad(self, bold_theta):
        """
        Compute the gradient of the classification loss evaluated
        at the given input parameter.

        Parameters
        ----------
        theta : ndarray of shape (K, d, d+1)
            Parameter of the model. For each class `k`, the first column
            of `bold_theta[k]` corresponds to the exogenous intensity parameter.
            The `d` next columns correspond to the interaction matrix.

        Returns :
        -------
        ndarray of shape shape (K, d, d+1)
            The gradient of the classification loss.
        """
        # Call the base class logic
        bold_theta = super().grad(bold_theta)

        grad = np.zeros(bold_theta.shape, dtype=np.float64)
        for rep in range(self._n_repetitions):
            F, grad_F = self._class_log_likelihoods(self._X[rep], self._end_time, bold_theta, self._decay, with_grad=True)
            pi = self._posterior(F, self._weights)
            y = self._y[rep]
            # d/dF_k of 4 * sum_m (pi_m - 1{m=y})^2 is
            # 8 pi_k (pi_k - 1{k=y} - ||pi||^2 + pi_y)
            coef = pi * (pi - np.sum(pi**2) + pi[y])
            coef[y] -= pi[y]
            grad += 8 * coef[:, None, None] * grad_F
        return grad / self._n_repetitions

    def hessian(self):
        raise NotImplementedError("Method hessian is not implemented for Hawkes model with log-likelihood goodness-of-fit functional")

    def lipschitz_const(self):
        raise NotImplementedError("The log-likelihood goodness-of-fit functional do not have a Lipschitz continuous gradient")

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Model", "Classification of Linear Hawkes process"],
                 ["Kernel function", "Exponential"],
                 ["Goodness-of-fit", "Log-likelihood"],
                 ["Gradient Lipchitz", "No"],
                 ["Network dimension", self._n_components],
                 ["Number of repetitions", self._n_repetitions],
                 ["Number of classes", self._n_classes],
                 ["Observation uppper-bound", self._end_time],
                 ["Decay of the kernel", self._decay]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def __repr__(self):
        return super().__repr__()
