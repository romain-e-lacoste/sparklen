# Author: Romain E. Lacoste
# License: BSD-3-Clause

from tabulate import tabulate

from sparklen.prox.base.prox import Prox
from sparklen.prox.build.prox import ProxZero as CppProxZero


class ProxZero(Prox):
    """
    Proximal operator class for the identity.

    This class implements the proximal operator for the identity,
    which effectively results in no regularization being applied.

    Parameters
    ----------
    positive : bool, default=True
        Determines whether the penalty should be coupled together with
        a projection onto the set of vectors with non-negative entries.
        If `positive=True`, applies a constraint that enforces the entries
        to be non-negative, in addition to the penalty from the L2 norm.

    Attributes
    ----------
    pen_const : float
        The penalty constant that controls the strength of the
        regularization applied. This is a read-only property.

    start : int
        Start column index on which the proximal operator is applied.
        This is a read-only property.

    end : int
        End column index on which the proximal operator is applied.
        This is a read-only property.

    positive : bool
        If the penalty if coupled with a constraint that enforces the entries
        to be non-negative. If `positive=True`, applies a constraint that
        enforces the entries to be non-negative.
        This is a read-only property.
    """
    def __init__(self, positive=True):
        # Call the initializer of the base class Prox
        super().__init__(positive)
        self._cpp_prox = None

    def set_pen_const(self, pen_const):
        """ Set the object with the associated penalty constant """
        # Call the base class logic
        super().set_pen_const(pen_const)

    def set_application_range(self, start, end):
        """ Set the object with application range """
        # Call the base class logic
        super().set_application_range(start, end)

    def apply(self, x, step_size):
        """
        Apply the proximal operator to a point at the current iteration
        of the descent.

        Parameters
        ----------
        x : ndarray
            The current point on which the proximal operator will be applied.

        step_size : float
            The step size used in the current iteration of the descent.

        Returns
        -------
        output : ndarray
            The point resulting from the application of the proximal operator.
        """
        # Call the base class logic
        super().apply(x, step_size)

        self._cpp_prox = CppProxZero(self._pen_const, self._start, self._end, self._positive)

        return self._cpp_prox.apply(x, step_size)

    def print_info(self):
        """ Display information about the instantiated model object. """
        table = [["Proximal operator", "Identity"],
                 ["Induced regularization", "None"],
                 ["Penalization constant", self._pen_const],
                 ["Start of apply range", self._start],
                 ["End of apply range", self._end],
                 ["Positivity constraint", self._positive]]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def __repr__(self):
        return super().__repr__()
