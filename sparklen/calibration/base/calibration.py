# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod

from sparklen.hawkes.model import (
    ModelHawkesExpLeastSquares,
    ModelHawkesExpLogLikelihood,
)
from sparklen.optim.lr import BacktrackingLineSearchLR, LipschitzLR
from sparklen.optim.optimizer import AGD, GD
from sparklen.prox import ProxElasticNet, ProxL1, ProxL2, ProxZero


class Calibration(ABC):

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

    _optimizers = {
        "gd" : GD,
        "agd" : AGD
    }

    _lr_schedulers = {
        "lipschitz" : LipschitzLR,
        "backtracking" : BacktrackingLineSearchLR
    }

    def __init__(self, loss="least-squares", penalty="lasso", optimizer="agd", lr_scheduler="backtracking", max_iter=100, tol=1e-5, penalty_mu=False, verbose_bar=True, verbose=True):

        if loss not in self._losses:
            raise ValueError(f"The choosen loss, '{loss}', is not available for Hawkes model with exponential kernel. Choose instead from {list(self._losses.keys())}.")
        self._str_loss = loss

        if penalty not in self._penalties:
            raise ValueError(f"The choosen penalty, '{penalty}', is not available. Choose instead from {list(self._penalties.keys())}.")
        self._prox = self._penalties[penalty]()

        if optimizer not in self._optimizers:
            raise ValueError(f"The choosen optimizer, '{optimizer}', is not available. Choose instead from {list(self._optimizers.keys())}.")
        self._optimizer = self._optimizers[optimizer](lr_scheduler, max_iter, tol, verbose_bar=False, verbose=False, print_every=1, record_every=10)

        self._penalty_mu = penalty_mu
        self._verbose_bar = verbose_bar
        self._verbose = verbose

        self._best_kappa = None
        self._best_score = None

    @abstractmethod
    def calibrate(self, decay, data, end_time=None, grid_max=2.0, grid_step=0.1, refinement=True, refined_grid_step=0.01):
        # This part must be implemented in the subclasses method
        pass

    @property
    def best_kappa(self):
        return self._best_kappa

    @property
    def best_score(self):
        return self._best_score

    @abstractmethod
    def print_info(self):
        pass
