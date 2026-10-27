# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np


class ModelHawkesClassification(ABC):
    """
    Abstract class intended as a base class for Hawkes classification models.

    This class defines a core structure for Hawkes classification models
    classes through inheritance. Thus, it is intended for development
    purposes, not for end-users.
    """

    def __init__(self, decay=None, weights=None):

        # Initialize attribute with default value to call the setter
        self._X = None
        self._y = None
        self._end_time = None
        self._decay = None
        self._weights = None

        if decay is not None:
            self.decay = decay

        if weights is not None:
            self._weights = weights

        self._is_data_setted = False

    def check_set_state(self):
        if not self._is_data_setted:
            if self._X is None:
                raise AttributeError("The training data must have been set to run this computation. You must call set_data() before calling loss().")
            if self._y is None:
                raise AttributeError("The training labels must have been set to run this computation. You must call set_data() before calling loss().")
            if self._end_time is None:
                raise AttributeError("The observation end-time must have been set to run this computation. You must call set_data() before calling loss().")
        if self._decay is None:
            raise AttributeError("The kernel decay paremeter must have been set to run this computation")

    def set_data(self, X, y, end_time=None):
        # Check training data form
        if not len(X) >= 1:
            raise ValueError("The training data should have at least one repetition")
        self._n_repetitions = len(X)

        if not len(X[0]) >= 1:
            raise ValueError("The dimension of the network should be at least one")
        self._n_components = len(X[0])

        for index, sublist in enumerate(X):
            if len(sublist) != self._n_components:
                raise ValueError(f"The inner length of the {index}-th repetition of the training data should match the number of components {self._n_components}, "
                          f"but got {len(sublist)} instead.")
        if self._X is not None:
            warn("The training data has already been set. This will overwrite the existing one.", UserWarning)
        self._X = X

        # Check training labels form
        if not len(y) == self._n_repetitions:
            raise ValueError("The training labels should have the same length as the training data")
        self._n_classes = np.max(y)+1
        self._y = y

       # Check end_time form
        if end_time is None:
            self._end_time = max([max((np.max(events, initial=0.0) for events in inner), default=0.0) for inner in X])
        else:
            if end_time < 0:
                raise ValueError("The upper bound of observation should be positive")
            if self._end_time is not None:
                warn("The observation end-time has already been set. This will overwrite the existing one.", UserWarning)
            self._end_time = end_time

        self._is_setted = True

    def n_classes(self):
        return self._n_classes

    def n_repetitions(self):
        return self._n_repetitions

    def n_components(self):
        return self._n_components

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def end_time(self):
        return self._end_time

    @property
    def decay(self):
        return self._decay

    @property
    def weights(self):
        return self._weights

    @decay.setter
    def decay(self, decay):
        if decay < 0:
            raise ValueError("The decay parameter of the kernel should be positive")
        if self._decay is not None:
            warn("The decay parameter of the kernel has already been set. This will overwrite the existing one.", UserWarning)
        self._decay = decay

    @weights.setter
    def weights(self, weights):
        if self._weights is not None:
                warn("The probability of mixture has already been set. This will overwrite the existing one.", UserWarning)
        self._weights = weights

    @abstractmethod
    def loss(self, bold_theta):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the value of the loss evaluated at a given point.

        Parameters
        ----------
        theta : ndarray of shape (K, d, d+1)
            Parameter of the model which takes the form of a 3D array
            of shape (K, d, d+1). For each class, the first column
            corresponds to the exogenous intensity parameter.
            The d next columns correspond to the interaction matrix.

        Returns :
        -------
        'float'
            The value of the loss evaluated at the input parameter.
        """

        # Partially implemented logic in the abstract method
        self.check_set_state()
        if not isinstance(bold_theta, np.ndarray):
            raise TypeError(f"Input theta parameter should be a NumPy Array, but got {type(bold_theta).__name__} instead.")
        if bold_theta.shape != (self._n_classes, self._n_components, self._n_components+1):
            raise ValueError(f"Input theta parameter should be of shape {(self._n_classes, self._n_components, self._n_components+1)}, but got {bold_theta.shape} instead.")

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def grad(self, bold_theta):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the gradient of the loss evaluated at a given point.

        Parameters
        ----------
        theta : ndarray of shape (K, d, d+1)
            Parameter of the model which takes the form of a 3D array
            of shape (K, d, d+1). For each class, the first column
            corresponds to the exogenous intensity parameter.
            The M next columns correspond to the interaction matrix.

        Returns :
        -------
        ndarray of shape (K, d, d+1)
            The gradient of the loss evaluated at the input parameter.
        """

        # Partially implemented logic in the abstract method
        self.check_set_state()
        if not isinstance(bold_theta, np.ndarray):
            raise TypeError(f"Input theta parameter should be a NumPy Array, but got {type(bold_theta).__name__} instead.")
        if bold_theta.shape != (self._n_classes, self._n_components, self._n_components+1):
            raise ValueError(f"Input theta parameter should be of shape {(self._n_classes, self._n_components, self._n_components+1)}, but got {bold_theta.shape} instead.")

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def hessian(self):
        """
        Abstract method, must be implemented in subclasses that inherit it.

        Compute the hessian of the loss evaluated at a given point.

        Returns :
        -------
        'numpy.array'
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
        'float'
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
        return (f"{class_name}(decay={self._decay}, weights={self._weights})")
