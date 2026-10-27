# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np


class ModelHawkes(ABC):
    """
    Abstract class intended as a base class for Hawkes models.

    This class defines a core structure for Hawkes models classes
    through inheritance. Thus, it is intended for development purposes,
    not for end-users.
    """

    def __init__(self, decay=None):

        # Initialize attribute with default value to call the setter
        self._data = None
        self._end_time = None
        self._decay = None

        self._is_data_setted = False
        self._is_decay_setted = False

        if decay is not None:
            self.decay = decay
            self._is_decay_setted = True

    def check_set_state(self):
        if not self._is_data_setted:
            if self._data is None:
                raise AttributeError("The data must have been set to the ModelHawkes object to run this computation. You must call set_data() before calling loss().")
            if self._end_time is None:
                raise AttributeError("The observation end-time must have been set to the ModelHawkes object to run this computation. You must call set_data() before calling loss().")
        if not self._is_decay_setted:
            raise AttributeError("The kernel decay parameter must have been set to the ModelHawkes object to run this computation")

    def set_data(self, data, end_time=None):
        # Check data form
        if not len(data) >= 1:
            raise ValueError("The data should have at least one repetition")
        self._n_repetitions = len(data)
        if not len(data[0]) >= 1:
            raise ValueError("The dimension of the network should be at least one")
        self._n_components = len(data[0])
        for index, sublist in enumerate(data):
            if len(sublist) != self._n_components:
                raise ValueError(f"The inner length of the {index}-th repetition of data should match the number of components {self._n_components}, "
                          f"but got {len(sublist)} instead.")
        if self._data is not None:
            warn("The data has already been set. This will overwrite the existing one.", UserWarning)
        self._data = data
        # check end_time form
        if end_time is None:
            self._end_time = max([max((np.max(events, initial=0.0) for events in inner), default=0.0) for inner in data])
        else:
            if end_time < 0:
                raise ValueError("The upper bound of observation should be positive")
            if self._end_time is not None:
                warn("The observation end-time has already been set. This will overwrite the existing one.", UserWarning)
            self._end_time = end_time
        self._is_data_setted = True

    def n_repetitions(self):
        return self._n_repetitions

    def n_components(self):
        return self._n_components

    @property
    def data(self):
        return self._data

    @property
    def end_time(self):
        return self._end_time

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, decay):
        if decay < 0:
            raise ValueError("The decay parameter of the kernel should be positive")
        if self._decay is not None:
            warn("The decay parameter of the kernel has already been set. This will overwrite the existing one.", UserWarning)
        self._decay = decay
        self._is_decay_setted = True

    @abstractmethod
    def loss(self, theta):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the value of the loss evaluated at a given point.

        Parameters
        ----------
        theta : ndarray of shape (d, d+1)
            Parameter of the model. The first column corresponds to
            the exogenous intensity parameter. The d next columns
            correspond to the interaction matrix.

        Returns :
        -------
        float
            The value of the loss evaluated at the input parameter.
        """

        # Partially implemented logic in the abstract method
        self.check_set_state()
        if not isinstance(theta, np.ndarray):
            raise TypeError(f"Input theta parameter should be a NumPy Array, but got {type(theta).__name__} instead.")
        if theta.shape != (self._n_components, self._n_components+1):
            raise ValueError(f"Input theta parameter should be of shape {(self._n_components, self._n_components+1)}, but got {theta.shape} instead.")

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def grad(self, theta):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the gradient of the loss evaluated at a given point.

        Parameters
        ----------
        theta : ndarray of shape (d, d+1)
            Parameter of the model. The first column corresponds to
            the exogenous intensity parameter. The d next columns
            correspond to the interaction matrix.

        Returns :
        -------
        ndarray
            The gradient of the loss evaluated at the input parameter.
        """

        # Partially implemented logic in the abstract method
        self.check_set_state()
        if not isinstance(theta, np.ndarray):
            raise TypeError(f"Input theta parameter should be a NumPy Array, but got {type(theta).__name__} instead.")
        if theta.shape != (self._n_components, self._n_components+1):
            raise ValueError(f"Input theta parameter should be of shape {(self._n_components, self._n_components+1)}, but got {theta.shape} instead.")

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def hessian(self):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the hessian of the loss.

        Returns :
        -------
        ndarray
            The hessian matrix of the loss evaluated at the input parameter.

        Raises:
        -------
            NotImplementedError: If the method is not implemented by a subclass.
        """

        # Partially implemented logic in the abstract method
        self.check_set_state()

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def lipschitz_const(self):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the Lipschitz constant of the gradient.

        Returns :
        -------
        float
            The value of the Lipschitz constant.

        Raises:
        -------
            NotImplementedError: If the method is not implemented by a subclass.
        """

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def print_info(self):
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        return (f"{class_name}(decay={self._decay})")
