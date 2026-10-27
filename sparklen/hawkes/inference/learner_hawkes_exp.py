# Author: Romain E. Lacoste
# License: BSD-3-Clause

import numpy as np
from sklearn.base import BaseEstimator
from tabulate import tabulate

from sparklen.calibration import CalibrationCV, CalibrationEBIC
from sparklen.hawkes.model import (
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)
from sparklen.optim.lr import (
    BacktrackingLineSearchLR,
    LipschitzLR,
    TwoWayBacktrackingLineSearchLR,
)
from sparklen.optim.optimizer import AGD, GD
from sparklen.plot import plot_support, plot_values
from sparklen.prox import ProxElasticNet, ProxL1, ProxL2, ProxZero


class LearnerHawkesExp(BaseEstimator):
    """
    Learner class for Hawkes process with exponential kernel.

    This class performs inference for a Hawkes process based on observed data.

    The observed data consist of repeated realizations of a Hawkes process.

    .. math::
        \\left\\{\\left\\{t_{j,\\ell}^{(i)}\\right\\}_{\\ell \\in [N_j(T)]},
        \ j \\in [d], \ i \\in [n] \\right\}

    where

    * :math:`n \\geq 1` is the number of repetitions
    * :math:`d \\geq 1` is the number of components
    * :math:`T > 0` is the time horizon
    * :math:`t_{j,\\ell}^{(i)}` is the :math:`\\ell`-th event time of component :math:`j` in the :math:`i`-th repetition

    Given the observation of data, the strategy consist in minimizing
    the following optimization problem:

    .. math::
        \\hat{\\theta}_n(\\kappa) \\in \\arg\\min_{
        \\theta \in \\mathbb{R}^{d \\times d+1}}
        \\left\\{F_{T, n}(\\theta) + \\kappa \\Omega(\\theta) \\right\}

    where

    * :math:`F_{T, n}(\cdot)` is a loss function
    * :math:`F_{T, n}(\cdot)` is a regularization function
    * :math:`\\kappa` is the regularization constant

    This class implements this strategy and offers various options
    for loss functions, regularization functions, penalty constants,
    and optimization methods. The user can customize the estimation procedure
    through intuitive string-based arguments and configuration dictionaries.
    This class is designed as a `scikit-learn`-compatible estimator,
    enabling interaction with its model selection tools.

    Parameters
    ----------
    decay : float
        The decay hyperparameter of the exponential kernel of the process.
        This scalar dictates how quick the influences vanish over time.

    loss : str, {'least-squares', 'log-likelihood'}, default='least-squares'
        Specifies the loss function to be used. The available options are:

        - 'least-squares': The least-squares loss function.
        - 'log-likelihood': The log-likelihood loss function.

    penalty : str, {'none', 'lasso', 'ridge', 'elasticnet'}, default='none'
        Specifies the type of penalty to be applied. The possible options are:

        - 'none': No regularization
        - 'lasso': Lasso regularization (:math:`\ell_1`-penalty)
        - 'ridge': Ridge (or Tikhonov) regularization (:math:`\ell_2`-penalty)
        - 'elasticnet': Elastic-Net regularization (:math:`\ell_1 + \ell_2`-penalty).

    kappa_choice : str, {'cv', 'bic', 'ebic'}, default='ebic'
        Specifies the method for tuning the penalty constant. The available options are:

        - 'cv' : Cross-validation
        - 'bic' : Bayesian Information Criterion (BIC)
        - 'ebic' : Extended Bayesian Information Criterion (EBIC)

        If `penalty='none'`, no calibration is applied, and the corresponding
        entry in `kappa_choices` is effectively ignored.

    optimizer : str, {'gd', 'agd'}, default='agd'
        Specifies the optimization algorithm to use. The possible options are:

        - 'gd' : Gradient Descent (GD)
        - 'agd' : Accelerated Gradient Descent (AGD)

    lr_scheduler : str, {'lipschitz', 'backtracking'}, default='backtracking'
        Specifies the learning rate scheduler. The available options are:

        - 'lipschitz' : Lipschitz-based step size, usable only if `loss='least-squares'`
        - 'backtracking' : Backtracking line-search-based step size.
        - 'fast-backtracking' : Two-Way Backtracking line-search-based step size.

    max_iter : int, default=100
        The maximum number of iterations allowed during the optimization process.

    tol : float, default=1e-5
        The tolerance for the precision achieved during the optimization process.
        The optimization will stop when the convergence criterion falls below this value.
        If the tolerance is not reached, the optimizer will perform a maximum of `max_iter` iterations.

    penalty_mu : bool, default=False
        Determines whether the penalty is applied to `mu` (the first column of the parameter).
        If `penalty_mu=False`, the regularization is applied only to the interaction matrix.

    cv : int, default=5
        The number of cross-validation splits.
        This parameter is only used if `kappa_choice='cv'`.

    gamma : float, default=1.0
        Hyperparameter of the EBIC criterion.
        It should be between 0 and 1.
        If `gamma=0.0`, it corresponds to the classical BIC.
        Theoretical guarantees of consistency hold for values of `gamma` equal to 0.5 and 1.0.
        This parameter is only considered if `kappa_choice='ebic'`

    verbose_bar : bool, default=True
        Determines whether a progress bar is displayed during the optimization process,
        along with information such as the loss value and convergence criterion.

        - If `verbose_bar=False`, no information is displayed.
        - If set to `True`, details will be displayed every `print_every` iterations.

    verbose : bool, default=True
        Controls whether recorded information during the optimization phase is printed at the end.

        - If `verbose=False`, no information is printed.
        - If set to `True`, details will be displayed every `print_every` iterations.

    print_every : int, default=5
        Specifies the frequency at which history information is printed.
        Information will be printed when the iteration number is a multiple of `print_every`.

    record_every : int, default=5
        Specifies the frequency at which history information is recorded.
        Information will be recorded when the iteration number is a multiple of `record_every`.

    Attributes
    ----------
    estimated_params : ndarray of shape (d, d+1)
        The estimated parameters for this learner. The first column corresponds to
        the estimated exogenous intensity parameter, while the next `d` columns
        represent the estimated interaction matrix. This is a read-only property.

    Notes
    ----------
    This class handle the estimation of univariate Hawkes processes, as these
    are naturally included as a special case of the multivariate model.
    """

    _losses = {
        "least-squares" : ModelHawkesExpLeastSquares,
        "log-likelihood" : ModelHawkesExpLogLikelihood
    }

    _penalties = {
        "none" : ProxZero,
        "lasso" : ProxL1,
        "ridge" : ProxL2,
        "elasticnet" : ProxElasticNet
    }

    _kappa_choices = {
        "cv" : CalibrationCV,
        "bic" : CalibrationEBIC,
        "ebic" : CalibrationEBIC
    }

    _optimizers = {
        "gd" : GD,
        "agd" : AGD
    }

    _lr_schedulers = {
        "lipschitz" : LipschitzLR,
        "backtracking" : BacktrackingLineSearchLR,
        "fast-backtracking" : TwoWayBacktrackingLineSearchLR
    }

    def __init__(self, decay=None, loss="least-squares",
                 penalty="lasso", kappa_choice="ebic",
                 optimizer="agd", lr_scheduler="backtracking",
                 max_iter=100, tol=1e-5, penalty_mu=False,
                 cv=5, gamma=1.0, verbose_bar=True, verbose=True,
                 print_every=5, record_every=5):

        self._decay = decay

        if loss not in self._losses:
            raise ValueError(f"The choosen loss, '{loss}', is not available for Hawkes model with exponential kernel. Choose instead from {list(self._losses.keys())}.")
        self._model = self._losses[loss]()
        self._str_loss = loss

        if penalty not in self._penalties:
            raise ValueError(f"The choosen penalty, '{penalty}', is not available. Choose instead from {list(self._penalties.keys())}.")
        self._prox = self._penalties[penalty]()
        self._str_penalty = penalty

        if kappa_choice not in self._kappa_choices:
            raise ValueError(f"The choosen criteria to tune the penalization constant, '{kappa_choice}', is not available. Choose instead from {list(self._penalties.keys())}.")
        self._str_kappa_choice = kappa_choice
        if self._str_penalty != "none":
            if kappa_choice == "cv":
                self._calibration = self._kappa_choices[kappa_choice](cv, loss, penalty, optimizer, lr_scheduler, max_iter, tol, penalty_mu, verbose_bar, verbose)
            elif kappa_choice == "bic":
                self._calibration = self._kappa_choices[kappa_choice](0.0, loss, penalty, optimizer, lr_scheduler, max_iter, tol, penalty_mu, verbose_bar, verbose)
            elif kappa_choice == "ebic":
                self._calibration = self._kappa_choices[kappa_choice](gamma, loss, penalty, optimizer, lr_scheduler, max_iter, tol, penalty_mu, verbose_bar, verbose)
        else:
            self._calibration = None

        if optimizer not in self._optimizers:
            raise ValueError(f"The choosen optimizer, '{optimizer}', is not available. Choose instead from {list(self._optimizers.keys())}.")
        self._optimizer = self._optimizers[optimizer](lr_scheduler, max_iter, tol, verbose_bar, verbose, print_every, record_every)
        self._str_optimizer = optimizer

        self._str_lr_scheduler = lr_scheduler

        self._max_iter = max_iter
        self._tol = tol

        self._penalty_mu = penalty_mu

        self._cv = cv
        self._gamma = gamma

        self._verbose_bar = verbose_bar
        self._verbose = verbose
        self._print_every = print_every
        self._record_every = record_every

        self._best_kappa = None
        self._estimated_params = None
        self._is_fitted = False


    def fit(self, X, end_time=None):
        """
        Fit the Hawkes model to the given training data.

        Parameters
        ----------
        X : list of list of ndarray
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

        Returns
        -------
        self : object
            The instance of the fitted model.
        """
        if self._decay is None:
            raise AttributeError("The kernel decay parameter must have been given to call fit()")
        self._model.decay = self._decay
        # We set data to model
        self._model.set_data(X, end_time)

        if self._penalty_mu:
            self._prox.set_application_range(0, self._model.n_components()+1)
        else:
            self._prox.set_application_range(1, self._model.n_components()+1)

        # We tune kappa according to the chosen criteria
        if self._str_penalty != "none":
            self._calibration.calibrate(self._decay, X, end_time)
            self._best_kappa = self._calibration.best_kappa
        else:
            self._best_kappa = 0.0

        # We perform optimization with best kappa
        x0 = np.ones((self._model.n_components(), self._model.n_components()+1))*0.2
        self._prox.set_pen_const(self._best_kappa)
        self._optimizer.set_model(self._model)
        self._optimizer.set_prox(self._prox)
        self._optimizer.optimize(x0)
        self._estimated_params = self._optimizer.minimizer

        self._is_fitted = True

    @property
    def estimated_params(self):
        if not self._is_fitted:
                raise ValueError("Estimation has not been completed. You must call fit() before getting the estimated parameters.")
        return self._estimated_params

    def get_params(self, deep=True):
        """
        Get parameters for this learner.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "decay": self._decay,
            "loss": self._str_loss,
            "penalty": self._str_penalty,
            "kappa_choice": self._str_kappa_choice,
            "optimizer": self._str_optimizer,
            "lr_scheduler": self._str_lr_scheduler,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "penalty_mu": self._penalty_mu,
            "cv": self._cv,
            "gamma": self._gamma,
            "verbose_bar": self._verbose_bar,
            "verbose": self._verbose,
            "print_every": self._print_every,
            "record_every": self._record_every
        }
        return params

    def score(self, X, end_time=None):
        """
        Return the negative value of the loss function evaluated at the
        estimated parameters, based on the given data.

        This method requires that the `fit` method has been called
        beforehand to estimate the model parameters.

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
              across all components and repetitions. .

        Returns
        -------
        float
            The value of the loss function, computed based on the given data
            and evaluated at the estimated parameters.
        """
        if not self._is_fitted:
                raise ValueError("Estimation has not been completed. You must call fit() before calling score().")

        model_test = self._losses[self._str_loss](self._decay)
        model_test.set_data(X, end_time)

        return -model_test.loss(self._estimated_params)

    def plot_estimated_values(self, save_path=None, save_format='png', dpi=300, use_latex=False):
        """
        Plot the estimated parameter values as heatmaps.

        Both the estimated exogenous intensity and interaction matrix values are displayed,
        each with its own colorbar for better visualization.

        This method requires that the `fit` method has been called
        beforehand to estimate the model parameters.

        This method calls the plot function :function:`~sparklen.plot.plot_hawkes.plot_values`

        Parameters
        ----------
        save_path : str, optional, default=None
            The path where the plot will be saved. If not provided, the plot will not be saved.
        save_format : str, optional, default='png'
            The format in which to save the plot (e.g., 'png', 'pdf', 'pgf').
        dpi : int, optional, default=300
            The resolution of the saved plot. Higher values result in higher quality.
        use_latex : bool, optional, default=False
            Whether to use LaTeX for rendering text in the plot.
            If `True`, text will be rendered using LaTeX formatting.
        """
        plot_values(self._estimated_params, save_path, save_format, dpi, use_latex)

    def plot_estimated_support(self, save_path=None, save_format='png', dpi=300, use_latex=False):
        """
        Plot the support of the estimated parameters as heatmaps.

        Both the estimated exogenous intensity and the interaction matrix support are displayed,
        with zero and non-zero values highlighted in different colors.

        This method requires that the `fit` method has been called
        beforehand to estimate the model parameters.

        This method calls the plot function :function:`~sparklen.plot.plot_hawkes.plot_support`

        Parameters
        ----------
        save_path : str, optional, default=None
            The path where the plot will be saved. If not provided, the plot will not be saved.
        save_format : str, optional, default='png'
            The format in which to save the plot (e.g., 'png', 'pdf', 'pgf').
        dpi : int, optional, default=300
            The resolution of the saved plot. Higher values result in higher quality.
        use_latex : bool, optional, default=False
            Whether to use LaTeX for rendering text in the plot.
            If `True`, text will be rendered using LaTeX formatting.
        """
        plot_support(self._estimated_params, save_path, save_format, dpi, use_latex)

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Learner", "Hawkes Exponential"],
                 ["Decay", self._decay],
                 ["Loss", self._str_loss],
                 ["Penalty", self._str_penalty],
                 ["Kappa choice", self._str_kappa_choice],
                 ["Optimizer", self._str_optimizer],
                 ["Lr scheduler", self._str_lr_scheduler],
                 ["Maximum iterations", self._max_iter],
                 ["Tolerance", self._tol]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def __repr__(self):
        class_name = self.__class__.__name__
        return (f"{class_name}(decay={self._decay}, loss='{self._str_loss}', "
                f"penalty='{self._str_penalty}', kappa_choice='{self._str_kappa_choice}', "
                f"optimizer='{self._str_optimizer}', lr_scheduler='{self._str_lr_scheduler}', "
                f"max_iter={self._max_iter}, tol={self._tol}, penalty_mu={self._penalty_mu}, "
                f"cv={self._cv}, gamma={self._gamma}, verbose_bar={self._verbose_bar}, "
                f"verbose={self._verbose}, print_every={self._print_every}, "
                f"record_every={self._record_every})")

    def set_params(self, **params):
        """
        Set the decay parameter of this learner.

        This method allows compatibility with scikit-learn routines like
        `GridSearchCV` and `cross_validate` by enabling hyperparameter tuning.

        Parameters
        ----------
        **params : dict
            Learner decay parameter.

        Returns
        -------
        self : object
            Learner instance with updated decay parameter.
        """
        for key, value in params.items():
            if key == "decay":
                self._decay = value
            else:
                raise ValueError(f"Unsupported parameter: {key}")
        return self
