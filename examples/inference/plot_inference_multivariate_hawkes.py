"""
===============================================================================
Inference of a Multivariate Hawkes Process (MHP)
===============================================================================

The following provides a comprehensive guide to using
:class:`~sparklen.hawkes.inference.LearnerHawkesExp` to estimate a MHP.
"""

# Author: Romain E. Lacoste
# License: BSD-3-Clause

# Setup environment -----------------------------------------------------------

import numpy as np

from sparklen.hawkes.inference import LearnerHawkesExp
from sparklen.hawkes.simulation import SimuHawkesExp
from sparklen.plot import plot_values

# Set the true coefficients ---------------------------------------------------

d = 5
beta = 3.0

mu = np.array([0.6, 0.55, 0.6, 0.55, 0.6])

alpha = np.zeros((d,d))
alpha[:4, :4] += 0.1
alpha[2:, 2:] += 0.15

theta_star = np.hstack([np.reshape(mu, (d,-1)), alpha])

# Plot the true coefficients --------------------------------------------------

plot_values(theta_star)

# Simulate training data ------------------------------------------------------

T = 5.0
n = 1000

hawkes = SimuHawkesExp(
    mu=mu, alpha=alpha, beta=beta,
    end_time=T, n_samples=n,
    random_state=4)

hawkes.simulate()
data = hawkes.timestamps

# Perform estimation ----------------------------------------------------------

learner = LearnerHawkesExp(
    decay=beta, loss="least-squares", penalty="none",
    optimizer="agd", lr_scheduler="fast-backtracking",
    max_iter=200, tol=1e-5,
    verbose_bar=True, verbose=True,
    print_every=10, record_every=10)

learner.fit(X=data, end_time=T)

print(learner.estimated_params)

print(learner.score(X=data, end_time=T))

# Plot the estimated coefficients ---------------------------------------------

learner.plot_estimated_values()
