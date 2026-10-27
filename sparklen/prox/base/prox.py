# Author: Romain E. Lacoste
# License: BSD-3-Clause

from abc import ABC, abstractmethod

import numpy as np


class Prox(ABC):

    def __init__(self, positive=True):

        if not isinstance(positive, bool):
            raise ValueError("The positivity constraint input should be a boolean.")
        self._positive = positive

        self._pen_const = None
        self._start = None
        self._end = None

        self._is_pen_const_setted = False
        self._is_application_range_setted = False

    def check_set_state(self):
        if not self._is_pen_const_setted:
            raise AttributeError("The penalization constant must have been set to Prox object to run this computation")
        if not self._is_application_range_setted:
            if self._start is None:
                raise AttributeError("The start of apply range must have been set to Prox object to run this computation")
            if self._end is None:
                    raise AttributeError("The end of apply range must have been set to Prox object to run this computation")

    def set_pen_const(self, pen_const):
        if not pen_const >= 0:
            raise ValueError("The penalization constant should be non-negative.")
        self._pen_const = pen_const
        self._is_pen_const_setted = True

    def set_application_range(self, start, end):
        if not start >= 0:
            raise ValueError("The start of apply range should begin from zero.")
        self._start = start
        if not end >= 0:
            raise ValueError("The end of apply range should begin from zero.")
        self._end = end
        self._is_application_range_setted = True

    @property
    def pen_const(self):
        return self._pen_const

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def positive(self):
        return self._positive

    @abstractmethod
    def apply(self, x, step_size):
        # Partially implemented logic in the abstract method
        if not self._is_pen_const_setted:
            raise AttributeError("The penalization constant has not been set. You must call set_pen_const() before apply()")
        if not self._is_application_range_setted:
            raise AttributeError("The application range has not been set. You must call set_application_range() before apply()")

        if not isinstance(x, np.ndarray):
            raise TypeError(f"Input theta parameter should be a NumPy Array, but got {type(x).__name__} instead.")

        if not step_size >= 0:
            raise ValueError("The step size input should be positive.")

        # This part must be implemented in the subclasses method
        pass

    @abstractmethod
    def print_info(self):
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        return (f"{class_name}(positive={self._positive})")
