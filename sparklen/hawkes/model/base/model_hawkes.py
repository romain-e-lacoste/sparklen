# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np


def _validate_paths(data, end_time=None, name="data"):
    """
    Check repeated paths of a Hawkes process and return a safe copy.

    Every array of event times is converted to a C-contiguous float64 array
    (the layout read by the C++ core) and must be one-dimensional, finite,
    non-negative and sorted. With `end_time=None`, the end time is the
    largest event time; otherwise it must be positive and no event may
    exceed it.

    Returns
    -------
    paths : list of list of ndarray
        The converted paths.

    end_time : float
        The end time of the observation period.
    """
    if not len(data) >= 1:
        raise ValueError(f"The {name} should have at least one repetition")
    n_components = len(data[0])
    if not n_components >= 1:
        raise ValueError("The dimension of the network should be at least one")
    paths = []
    for i, path in enumerate(data):
        if len(path) != n_components:
            raise ValueError(f"The inner length of the {i}-th repetition of {name} should match the number of components {n_components}, "
                             f"but got {len(path)} instead.")
        converted = []
        for j, events in enumerate(path):
            events = np.ascontiguousarray(events, dtype=np.float64)
            if events.ndim != 1:
                raise ValueError(f"{name}[{i}][{j}] should be a one-dimensional array of event times, but has shape {events.shape}.")
            if not np.all(np.isfinite(events)):
                raise ValueError(f"{name}[{i}][{j}] contains NaN or infinite event times.")
            if np.any(np.diff(events) < 0):
                raise ValueError(f"The event times in {name}[{i}][{j}] should be sorted in increasing order.")
            if events.size and events[0] < 0:
                raise ValueError(f"{name}[{i}][{j}] contains negative event times.")
            converted.append(events)
        paths.append(converted)

    last_event = max((events[-1] for path in paths for events in path if events.size), default=0.0)
    if end_time is None:
        if not last_event > 0:
            raise ValueError(f"The observation end-time cannot be inferred from {name} without positive event times. Pass end_time explicitly.")
        end_time = last_event
    else:
        if not (np.isfinite(end_time) and end_time > 0):
            raise ValueError("The upper bound of observation should be positive")
        if last_event > end_time:
            raise ValueError(f"The largest event time ({last_event}) exceeds the upper bound of observation end_time={end_time}. "
                             "Remove the events after end_time or increase end_time.")
    return paths, float(end_time)


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
        # Check the data form, convert the event times and set the end time
        paths, checked_end_time = _validate_paths(data, end_time)
        self._n_repetitions = len(paths)
        self._n_components = len(paths[0])
        if self._data is not None:
            warn("The data has already been set. This will overwrite the existing one.", UserWarning)
        self._data = paths
        if end_time is not None and self._end_time is not None:
            warn("The observation end-time has already been set. This will overwrite the existing one.", UserWarning)
        self._end_time = checked_end_time
        self._is_data_setted = True
        self._invalidate_cache()

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
        self._invalidate_cache()

    def _invalidate_cache(self):
        """
        Invalidate the quantities precomputed from the data, the end time and
        the decay, so that they are recomputed at their next use. Called by
        set_data and the decay setter; subclasses that cache them override it.
        """
        pass

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

        # The C++ core reads theta as a packed row-major float64 buffer
        return np.ascontiguousarray(theta, dtype=np.float64)

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

        # The C++ core reads theta as a packed row-major float64 buffer
        return np.ascontiguousarray(theta, dtype=np.float64)

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
