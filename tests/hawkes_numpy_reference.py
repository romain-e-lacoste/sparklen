# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Test helpers, not tests: pure-NumPy reference implementations of the
Hawkes models, a finite-difference gradient and a generator of tiny paths,
shared by the test files of the subfolders.

The model implementations are written for clarity, not speed, and share no
code with the C++ core: intensities are evaluated directly from their
definition and integrals are computed by Gauss-Legendre quadrature between
consecutive events, where the intensity is smooth. They are only meant for
tiny paths.
"""

import numpy as np

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(30)


def intensity(path, t, decay, theta):
    """Evaluate the left-continuous intensity lambda(t) of one path, shape (d,): events at time t are excluded."""
    mu, alpha = theta[:, 0], theta[:, 1:]
    excitation = np.array([
        np.sum(decay * np.exp(-decay * (t - events[events < t])))
        for events in path
    ])
    return mu + alpha @ excitation


def _integrate(path, end_time, func):
    """Integrate func(t) on [0, end_time], split at every event time."""
    cuts = np.unique(np.concatenate([[0.0, end_time], *path]))
    cuts = cuts[(cuts >= 0) & (cuts <= end_time)]
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        half, mid = (b - a) / 2, (b + a) / 2
        for x, w in zip(_NODES, _WEIGHTS):
            total += half * w * func(mid + half * x)
    return total


def least_squares_loss(data, end_time, decay, theta):
    """Compute the least-squares contrast averaged over the repetitions."""
    losses = []
    for path in data:
        square = _integrate(
            path, end_time,
            lambda t: np.sum(intensity(path, t, decay, theta) ** 2))
        jumps = sum(intensity(path, t, decay, theta)[i]
                    for i, events in enumerate(path) for t in events)
        losses.append((square - 2 * jumps) / end_time)
    return np.mean(losses)


def log_likelihood(data, end_time, decay, theta):
    """Compute the log-likelihood log L summed over the repetitions (the model's loss is -log L)."""
    total = 0.0
    for path in data:
        compensator = _integrate(
            path, end_time,
            lambda t: np.sum(intensity(path, t, decay, theta)))
        log_jumps = sum(np.log(intensity(path, t, decay, theta)[i])
                        for i, events in enumerate(path) for t in events)
        total += log_jumps - compensator
    return total


def finite_difference_grad(func, x, step=1e-6):
    """Compute the central finite-difference gradient of a scalar function of an array."""
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        h = step * max(1.0, abs(x[index]))
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[index] += h
        x_minus[index] -= h
        grad[index] = (func(x_plus) - func(x_minus)) / (2 * h)
    return grad


def tiny_paths(n_paths=3, n_components=2, end_time=5.0, seed=0):
    """Generate a few short, sorted, float64 paths, with event times drawn uniformly on [0, end_time)."""
    rng = np.random.default_rng(seed)
    return [
        [np.sort(rng.uniform(0, end_time, rng.integers(3, 9)))
         for _ in range(n_components)]
        for _ in range(n_paths)
    ]
