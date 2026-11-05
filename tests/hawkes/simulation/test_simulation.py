# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Simulation of Hawkes processes."""

import numpy as np
import pytest

from sparklen.hawkes.simulation import SimuHawkesExp


def test_spectral_radius_is_the_largest_modulus():
    """Check that the spectral radius is a real float equal to the largest eigenvalue
    modulus, also when some eigenvalues are complex."""
    # A 3-cycle: eigenvalues 0.5 and 0.5 exp(+-2i pi/3), two of them complex
    alpha = 0.5 * np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    radius = SimuHawkesExp(np.full(3, 0.1), alpha, 1.0, 1.0, 1).spectral_radius()
    assert isinstance(radius, float)
    assert radius == pytest.approx(0.5)

    # Real eigenvalues 0.5 and 0.1
    alpha = np.array([[0.2, 0.3], [0.1, 0.4]])
    assert SimuHawkesExp(np.full(2, 0.1), alpha, 1.0, 1.0, 1).spectral_radius() == pytest.approx(0.5)


def test_unstable_process_is_not_simulated():
    """Check that a process with spectral radius at least 1
    is not simulated but raises a ValueError."""
    simu = SimuHawkesExp(np.array([0.5, 0.5]), np.array([[0.6, 0.5], [0.5, 0.6]]),
                         1.0, 10.0, 1, random_state=0)
    with pytest.raises(ValueError, match="spectral radius"):
        simu.simulate()


def test_simulate_returns_self_and_is_reproducible():
    """Check that simulate returns self and that a fixed random_state
    reproduces the paths."""
    args = (np.array([0.5, 0.3]), np.array([[0.3, 0.1], [0.0, 0.2]]), 2.0, 10.0, 3)
    first = SimuHawkesExp(*args, random_state=0)
    assert first.simulate() is first
    second = SimuHawkesExp(*args, random_state=0).simulate()
    assert len(first.timestamps) == len(second.timestamps) == 3
    for path_a, path_b in zip(first.timestamps, second.timestamps):
        for events_a, events_b in zip(path_a, path_b):
            np.testing.assert_array_equal(events_a, events_b)
