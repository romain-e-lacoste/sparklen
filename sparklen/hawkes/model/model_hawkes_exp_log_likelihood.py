# Author: Romain E. Lacoste
# License: BSD-3-Clause

from tabulate import tabulate

from sparklen.hawkes.model.base.model_hawkes import ModelHawkes
from sparklen.hawkes.model.build.hawkes_model import (
    ModelHawkesExpLogLikelihood as CppModelHawkesExpLogLikelihood,
)


class ModelHawkesExpLogLikelihood(ModelHawkes):
    """
    Model class for Hawkes process with exponential kernel and given decay.

    This class handles model-related calculations, including loss evaluation,
    gradient computation, and Hessian matrix calculation

    The observed data consist of repeated realizations of a Hawkes process.

    .. math::
        \\left\\{\\left\\{t_{j,\\ell}^{(i)}\\right\\}_{\\ell \\in [N_j(T)]},
        \ j \\in [d], \ i \\in [n] \\right\}

    where

    * :math:`n \\geq 1` is the number of repetitions
    * :math:`d \\geq 1` is the number of components
    * :math:`T > 0` is the time horizon
    * :math:`t_{j,\\ell}^{(i)}` is the :math:`\\ell`-th event time of component :math:`j` in the :math:`i`-th repetition


    The loss function used as goodness-of-fit is the log-likelihood
    averaged over the repetitions. For a given parameter
    :math:`\\theta=(\\mu, \\alpha)` it is given by:

    .. math::
        \\sum_{i=1}^n \\left(\\sum_{j=1}^d \\int_0^T \
        \\lambda_{j, \\theta}^{(i)}(t) \ \\textrm{d} t
        - \\sum_{j=1}^d\\sum_{\\ell : t_{j, \\ell}^{(i)} < T}
        \\log\\left(\\lambda_{j, \\theta}^{(i)}\\left(t_{j, \\ell}^{(i)}
        \\right)\\right)\\right)

    where :math:`\\lambda_{j, \\theta}(t)` is the intensity function:

    .. math::
        \\mu_j + \\sum_{j'=1}^d  \\alpha_{j,j'} \\sum_{\\ell : t_{j',\\ell} < t}
        \\beta e^{-\\beta(t-t_{j',\\ell})}

    where

    * :math:`(\\mu_j)_{j \\in [d]}` is the vector of exogenous intensities
    * :math:`(\\alpha_{j, j'})_{j, j' \\in [d]}` is the matrix of interactions
    * :math:`\\beta` is the fixed and given decay of the exponential kernel

    Parameters
    ----------
    decay : float, default=None
        The decay hyperparameter of the exponential kernel of the process.
        This scalar dictates how quick the influences vanish over time.

    Attributes
    ----------
    data : list of list of ndarray
        The data used to set up the model. This is a read-only property.

    end_time : float
        Time horizon used to set up the model. This is a read-only proterty.

    decay : float
        The decay parameter used in model calculations. This property can be modified.

    Notes
    ----------
    This class handle the calculations for univariate Hawkes processes, as these
    are naturally included as a special case of the multivariate model.

    See Also
    --------
    :class:`~sparklen.hawkes.model.ModelHawkesExpLeastSquares` : model class
    for least-squares loss.

    """
    def __init__(self, decay=None):
        # Call the initializer of the base class ModelHawkes
        super().__init__(decay)

    def set_data(self, data, end_time=None):
        """
        Initialize the model with data and a specified time horizon.

        Parameters
        ----------
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
        """
        # Call the base class logic
        super().set_data(data, end_time)
        self._cpp_model = CppModelHawkesExpLogLikelihood(self._n_repetitions, self._n_components)

    def loss(self, theta, neg=True):
        """
        Compute the value of the log-likelihood loss
        evaluated at the given input parameter.

        Parameters
        ----------
        theta : ndarray of shape (d, d+1)
            Parameter of the model. The first column corresponds to
            the exogenous intensity parameter. The `d` next columns
            correspond to the interaction matrix.

        neg : bool, default=True
            Determines whether to compute the negative log-likelihood.
            If `True`, computes the negative log-likelihood.
            If `False`, computes the standard log-likelihood.

        Returns :
        -------
        float
            The value of the log-likelihood loss.
        """
        # Call the base class logic
        super().loss(theta)
        return self._cpp_model.compute_averaged_loss(self._data, self._end_time, self._decay, theta, neg)

    def grad(self, theta, neg=True):
        """
        Compute the gradient of the log-likelihood loss
        evaluated at the given input parameter.

        Parameters
        ----------
        theta : ndarray of shape (d, d+1)
            Parameter of the model. The first column corresponds to
            the exogenous intensity parameter. The `d` next columns
            correspond to the interaction matrix.

        neg : bool, default=True
            Determines whether to compute the negative log-likelihood.
            If `True`, computes gradient of the negative log-likelihood.
            If `False`, computes gradient of the standard log-likelihood.

        Returns :
        -------
        ndarray of shape shape (d, d+1)
            The gradient of the log-likelihood loss.
        """
        # Call the base class logic
        super().grad(theta)
        return self._cpp_model.compute_averaged_grad(self._data, self._end_time, self._decay, theta, neg)

    def hessian(self):
        raise NotImplementedError("Method compute_hessian is not implemented for Hawkes model with log-likelihood goodness-of-fit functional")

    def lipschitz_const(self):
        raise NotImplementedError("The log-likelihood goodness-of-fit functional do not have a Lipschitz continuous gradient")

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Model", "Linear Hawkes"],
                 ["Kernel function", "Exponential"],
                 ["Goodness-of-fit", "Log-Likelihood"],
                 ["Gradient Lipchitz", "No"],
                 ["Network dimension", self._n_components],
                 ["Number of repetition", self._n_repetitions],
                 ["Observation uppper-bound", self._end_time],
                 ["Decay of the kernel", self._decay]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def __repr__(self):
        return super().__repr__()
