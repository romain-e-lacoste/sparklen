# Author: Romain E. Lacoste
# License: BSD-3-Clause

from warnings import warn

import numpy as np
from numpy.linalg import eig
from numpy.random import default_rng


class SimuHawkesExp():
    """
    Simulation class for Hawkes process with exponential kernel.

    This class generates data consisting of repeated realizations of a Hawkes process.

    The event simulation is based on the cluster representation of the process.

    A :math:`d`-dimensional Hawkes process, denoted :math:`N = (N_1, \dots, N_d)`
    is given by :math:`d` point processes on :math:`\mathbb{R}_+^*`, with
    :math:`d \geq 1` the dimension of the network.

    Each process :math:`N_j` is characterized by its intensity function
    :math:`\\lambda_j(t)`:

    .. math::
        \\mu_j + \\sum_{j'=1}^d  \\alpha_{j,j'} \\sum_{\\ell : t_{j',\\ell} < t}
        \\beta e^{-\\beta(t-t_{j',\\ell})}

    where

    * :math:`(\\mu_j)_{j \\in [d]}` is the vector of exogenous intensities
    * :math:`(\\alpha_{j, j'})_{j, j' \\in [d]}` is the matrix of interactions
    * :math:`\\beta` is the fixed and given decay of the exponential kernel

    Parameters
    ----------
    mu : ndarray of shape (d, )
        Exogenous intensitie of the process. Each `mu[j]` expresses the arrival
        of spontaneous events for the `j`-th process.

    alpha : ndarray of shape (d, d)
        Interaction matrix of the process. Each `alpha[j][j']` reflects the
        positive influence of the `j'`-th one-dimensional process on the `j`-th
        one-dimensional process.

    beta : float
        Common decay scalar of the process. Dictates how quick the influences
        vanish over time.

    end_time : float
        The end time of the observation period. The time horizon defines
        the interval `[0, T]` over which the Hawkes process is observed.

    n_samples : int
        The total number of repeated paths generated in the simulation.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.
        Pass an int for reproducible output across multiple function calls.


    Attributes
    ----------
    timestamps : list of list of ndarray
        The generated event timestamps. The outer list has length `n`,
        representing repetitions. Each inner list has length `d`, where
        each element is a one-dimensional ndarray containing the
        event times of a specific component. This property can be modified.

    Notes
    ----------
    This class handle the simulation of univariate Hawkes processes, as these
    are naturally included as a special case of the multivariate model.
    """

    def __init__(self, mu, alpha, beta, end_time, n_samples, random_state=None):

        self._check_param_form(mu, alpha, beta)
        self._mu = mu
        self._alpha = alpha
        self._beta = beta
        self._end_time = end_time
        self._n_components = mu.shape[0]

        if not n_samples >= 1:
            raise ValueError("There should be at least one repetition")
        self._n_samples = n_samples
        self._timestamps = None

        self._generator = default_rng(random_state)

    @property
    def timestamps(self):
        return self._timestamps

    @timestamps.setter
    def timestamps(self, timestamps):
        if not isinstance(timestamps, list):
            raise ValueError("The timestamps should be a list.")
        if self._timestamps is not None:
            warn("The timestamps of the process has already been set. This will overwrite the existing one.", UserWarning)
        self._timestamps = timestamps

    @staticmethod
    def _check_param_form(mu, alpha, beta):

        if not np.all(mu >= 0):
            raise ValueError("Exegenous intensity of the Hawkes process should be non-negative.")

        if not np.all(alpha >= 0):
            raise ValueError("Intensity of interaction of the Hawkes process should be non-negative.")

        if not beta >= 0:
            raise ValueError("The decay rate of the Hawkes process should be non-negative.")

    def simulate(self):
        """
        Simulate repeated paths of the Hawkes process
        and store the generated timestamps.

        Returns
        -------
        self : object
            The instance of the simulated object.
        """
        self._timestamps = []

        for _ in range(self._n_samples):
            path = self._simulate_single_path()
            self._timestamps.append(path)


    def _simulate_single_path(self):

        path = [[] for _ in range(self._n_components)]

        ancestor = [[] for _ in range(self._n_components)]
        # Simulation of the immigrants
        for j in range(self._n_components):
            k = self._generator.poisson(self._mu[j] * self._end_time)
            ancestor[j].extend(self._generator.uniform(0, self._end_time, k))
            path[j].extend(ancestor[j])


        # Simulation of the offsprings
        while (any(bool(sublist) for sublist in ancestor)):
            offsprings = [[] for _ in range(self._n_components)]
            for j in range(self._n_components):
                if bool(ancestor[j]):
                    for a_j_l in ancestor[j]:
                        for j2 in range(self._n_components):
                            if self._alpha[j2][j] > 0:
                                k = self._generator.poisson(self._alpha[j2][j])
                                offsprings_j2 = self._generator.exponential(1 / self._beta, k) + a_j_l
                                offsprings[j2].extend(offsprings_j2)
                                path[j2].extend(offsprings_j2)
            ancestor = offsprings

        # Remove elements above the upper bound and sort and convert each sublist to numpy array
        path[:] = [np.array(sorted([x for x in sublist if x < self._end_time])) for sublist in path]

        return path

    def compensator(self, t):
        """
        Placeholder for a future method that will compute the compensator of
        the Hawkes process.

        This method is not yet implemented.

        Parameters
        ----------
        t : float
            Time for which the compensator is evaluated.
        Returns
        -------
        None
            This function is not yet implemented.

        Raises
        ------
        NotImplementedError
            Always raised since the method is not implemented.
        """
        raise NotImplementedError("This method is not yet implemented.")

    def __repr__(self):
        class_name = self.__class__.__name__
        return (f"{class_name}(\n"
          f"  mu={self._mu},\n"
          f"  alpha={self._alpha},\n"
          f"  beta={self._beta},\n"
          f"  end_time={self._end_time},\n"
          f"  n_samples={self._n_samples},\n"
          f"  random_state={self._generator}\n"
          f")")

    def spectral_radius(self):
        """
        Compute and return the spectral radius of the interaction matrix.

        The spectral radius is defined as the largest absolute eigenvalue
        of the interaction matrix of the process.

        Returns :
        -------
        spectral_radius : float
            The spectral radius of the interaction matrix.
        """
        return np.max(eig(self._alpha)[0])
        # spectral_radius = np.max(eig(self._alpha)[0])
        # if spectral_radius > 1:
        #     warn("The spectral radius is greater than one.", UserWarning)
        # return spectral_radius
