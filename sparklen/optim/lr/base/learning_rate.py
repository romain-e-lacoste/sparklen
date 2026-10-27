# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):

    def __init__(self):
        self._step_size = None
        self._is_model_setted = False
        self._is_prox_setted = False

    # def step(self, *args, **kwargs):
    #     """
    #     Method to determine the step size for the current iteration.
    #     If the learning rate strategy is dynamic, it calculates the step size.
    #     Otherwise, it returns the precomputed static step size.
    #     """
    #     if self.is_dynamic:
    #         # Call the dynamic step size calculation method
    #         return self.calculate_step_size(*args, **kwargs)
    #     else:
    #         # Return the static step size
    #         return self._step_size

    def check_set_state(self):
        if not self._is_model_setted:
            raise AttributeError("The model has not been setted to the learning rate scheduler object. You must call set_model() before optimize().")
        if not self._is_prox_setted:
            raise AttributeError("The proximal operator has not been setted to the learning rate scheduler object. You must call set_prox() before optimize().")

    def set_model(self, model):
        model.check_set_state()
        self._model = model
        self._is_model_setted = True

    def set_prox(self, prox):
        prox.check_set_state
        self._prox = prox
        self._is_prox_setted = True

    @abstractmethod
    def step(self, search_point, loss_search_point, grad_search_point):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the step size based on the current state of the optimization state
        and compute new iterate and loss based on the latter.
        """
        self.check_set_state()
        pass

    @abstractmethod
    def print_info(self):
        pass
