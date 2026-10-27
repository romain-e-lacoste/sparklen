# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod

from tabulate import tabulate

from sparklen.optim.lr import (
    BacktrackingLineSearchLR,
    LipschitzLR,
    TwoWayBacktrackingLineSearchLR,
)


class Optimizer(ABC):

    _lr_schedulers = {
        "lipschitz" : LipschitzLR,
        "backtracking" : BacktrackingLineSearchLR,
        "fast-backtracking" : TwoWayBacktrackingLineSearchLR
    }

    def __init__(self, lr_scheduler, max_iter, tol, verbose_bar=True, verbose=True, print_every=10, record_every=1):

        if lr_scheduler not in self._lr_schedulers:
            raise ValueError(f"The choosen learning rate scheduler, '{lr_scheduler}', is not available. Choose instead from {list(self._lr_schedulers.keys())}.")
        self._lr_scheduler = self._lr_schedulers[lr_scheduler]()

        self._max_iter = max_iter
        self._tol = tol
        self._verbose_bar = verbose_bar
        self._verbose = verbose
        self._print_every = print_every
        self._record_every = record_every
        self._history = {
            "x": [],
            "loss": [],
            "grad": [],
            "lr": [],
            "rel_loss": [],
            "iter": []
        }
        self._minimizer = None
        self._elapsed_time = None

        self._is_model_setted = False
        self._is_prox_setted = False

        self._is_optimized = False

    def check_set_state(self):
        if not self._is_model_setted:
            raise AttributeError("The model has not been setted to the Optimizer object. You must call set_model() before optimize().")
        if not self._is_prox_setted:
            raise AttributeError("The proximal operator has not been setted to the Optimizer object. You must call set_prox() before optimize().")

    def set_model(self, model):
        model.check_set_state()
        self._model = model
        self._is_model_setted = True

    def set_prox(self, prox):
        prox.check_set_state
        self._prox = prox
        self._is_prox_setted = True

    @property
    def minimizer(self):
        if not self._is_optimized:
            raise ValueError("Optimization has not been completed. You must call optimize() before getting the minimizer.")
        return self._minimizer

    @property
    def elapsed_time(self):
        if not self._is_optimized:
            raise ValueError("Optimization has not been completed. You must call optimize() before getting the elapsed time.")
        return self._elapsed_time

    @property
    def history(self):
        if not self._is_optimized:
            raise ValueError("Optimization has not been completed. You must call optimize() before getting the history.")
        return self._history

    @abstractmethod
    def _initialize_values(self, x0):
        pass

    @abstractmethod
    def _step(self, x, *args):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Perform a single optimization step.
        """
        pass

    @abstractmethod
    def optimize(self, x0):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Perform the optimization task.
        """
        self.check_set_state()

        self._lr_scheduler.set_model(self._model)
        self._lr_scheduler.set_prox(self._prox)

        if self._is_optimized:

            # Resetting state
            self._history = {key: [] for key in self._history.keys()}  # Clear the history
            self._minimizer = None  # Clear the previous minimizer
            self._is_optimized = False  # Ensure this is reset

        # This part must be implemented in the subclasses method
        pass

    def print_history(self):
        if not self._is_optimized:
            raise RuntimeError("Optimization has not been completed. You must call optimize() before print_history().")

        # Extract relevant history information
        loss_history = self._history["loss"]
        lr_history = self._history["lr"]
        rel_loss_history = self._history["rel_loss"]
        iteration_history = self._history["iter"]

        # Combine them into a list of rows
        history_table = []
        for i in range(len(iteration_history)):
            history_table.append([iteration_history[i], loss_history[i], lr_history[i], rel_loss_history[i]])

        # Define headers for the table
        headers = ["Iter", "Loss", "Lr", "Tol"]

        # Print the table using tabulate
        print(tabulate(history_table, headers=headers, tablefmt="grid"))


    def record_history(self, x, loss_x, grad_x, step_size, rel_loss, iteration):
        self._history["x"].append(x)
        self._history["loss"].append(loss_x)
        self._history["grad"].append(grad_x)
        self._history["lr"].append(step_size)
        self._history["rel_loss"].append(rel_loss)
        self._history["iter"].append(iteration)

    @abstractmethod
    def print_info(self):
        pass
