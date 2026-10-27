# Author: Romain E. Lacoste
# License: BSD-3-Clause

import numpy as np
from numpy.random import default_rng

from sparklen.hawkes.simulation.simu_hawkes_exp import SimuHawkesExp


def make_classification(bold_mu, bold_alpha, beta, end_time, n_samples, n_classes, weights=None, random_state=None):
    """
    Generate a random `K`-class classification problem.

    This function generates data consisting of labeled repeated realizations of Hawkes processes.

    A :math:`d`-dimensional Hawkes process, denoted :math:`N = (N_1, \dots, N_d)`
    is given by :math:`d` point processes on :math:`\mathbb{R}_+^*`, with
    :math:`d \geq 1` the dimension of the network.

    Each process :math:`N_j` is characterized by its intensity function, which
    depends on :math:`Y` and is given by:

    .. math::
        \\mu_{Y,j} + \\sum_{j'=1}^d \\alpha_{Y,j,j'} \\int_0^t
        \\beta e^{-\\beta(t-s)} \ \\textrm{d} N_{j'}(s)

    where

    * :math:`(\\mu_{k,j})_{j \\in [d]}` is the vector of exogenous intensities of class `k`
    * :math:`(\\alpha_{k, j, j'})_{j, j' \\in [d]}` is the matrix of interactions of class `k`
    * :math:`\\beta` is the fixed and given decay of the exponential kernel

    Parameters
    ----------
    bold_mu : ndarray of shape (K, d, )
        Exogenous intensities of each class. For each class `k`, `mu[k][j]`
        expresses the arrival of spontaneous events for the `j`-th process.

    bold_alpha : ndarray of shape (K, d, d)
        Interaction matrix of the process. For each class `k`, `alpha[k][j][j']`
        reflects the positive influence of the `j'`-th one-dimensional process
        on the `j`-th one-dimensional process.

    beta : float
        Common decay scalar of the process. Dictates how quick the influences
        vanish over time.

    end_time : float
        The end time of the observation period. The time horizon defines
        the interval `[0, T]` over which the Hawkes process is observed.

    n_samples : int
        The total number of repeated paths generated in the simulation.

    n_classes : int
        The number of classes (or labels) of the classification problem.

    weights : ndarray of shape (K,), default=None
        The proportions of samples assigned to each class.
        If `weights=None`, then classes are balanced.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X : list of list of ndarray
        The generated samples. The outer list has length `n`,
        representing repetitions. Each inner list has length `d`, where
        each element is a one-dimensional ndarray containing the
        event times of a specific component.
    y : ndarray of shape (n,)
        The integer labels for class membership of each sample.

    """
    generator = default_rng(random_state)

    if weights is not None:
        if len(weights) is not n_classes:
            raise ValueError("The length of weights should match the number of classes.")
    else:
        weights = np.ones(n_classes)/n_classes

    y = generator.choice(n_classes, size=n_samples, p=weights)
    X = [None]*n_samples
    for k in range(n_classes):
        idx  = np.where(y == k)[0]
        for i in idx:
            hawkes = SimuHawkesExp(bold_mu[k], bold_alpha[k], beta, end_time, 1, generator)
            hawkes.simulate()
            X[i] = hawkes.timestamps[0]

    return X, y
