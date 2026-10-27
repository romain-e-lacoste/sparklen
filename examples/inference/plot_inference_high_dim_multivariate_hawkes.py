"""
===============================================================================
Inference of a sparse high-dimensional Multivariate Hawkes Process (MHP)
===============================================================================

We shift our focus to a high-dimensional MHP where the interaction matrix
exhibits a sparse structure. This example provides a detailed walkthrough for
using:class:`~sparklen.hawkes.inference.LearnerHawkesExp`, showcasing
the specialized options designed for such cases.
"""

# Author: Romain E. Lacoste
# License: BSD-3-Clause

# Setup environment -----------------------------------------------------------

import numpy as np

from sparklen.hawkes.inference import LearnerHawkesExp
from sparklen.hawkes.simulation import SimuHawkesExp
from sparklen.plot import plot_values

# Load the true coefficients --------------------------------------------------

d = 25
beta = 3.0

mu = np.ones(d)*0.4

alpha = np.zeros((d,d))
alpha[:3, :3] = 0.25
alpha[3:7, 3:7] = 0.2
alpha[7:10, 7:10] = 0.25
alpha[10:12, 10:12] = 0.2
alpha[12, 12] = 0.3
alpha[13:15, 13:15] = 0.2
alpha[15:18, 15:18] = 0.25
alpha[18:22, 18:22] = 0.2
alpha[22:, 22:] = 0.25

theta_star = np.hstack([np.reshape(mu, (d,-1)), alpha])

# Plot the true coefficients --------------------------------------------------

plot_values(theta_star)

# Simulate training data ------------------------------------------------------

T = 5.0
n = 250

hawkes = SimuHawkesExp(
    mu=mu, alpha=alpha, beta=beta,
    end_time=T, n_samples=n,
    random_state=8)

hawkes.simulate()
data = hawkes.timestamps

# Perform estimation ----------------------------------------------------------

learner_lasso_cv = LearnerHawkesExp(
    decay=beta, loss="least-squares",
    penalty="lasso", kappa_choice="cv",
    optimizer="agd", lr_scheduler="fast-backtracking",
    max_iter=200, tol=1e-5,
    penalty_mu=False, cv=10,
    verbose_bar=True, verbose=True,
    print_every=10, record_every=10)

learner_lasso_cv.fit(X=data, end_time=T)
theta_hat_lasso_cv = learner_lasso_cv.estimated_params


learner_lasso_ebic = LearnerHawkesExp(
    decay=beta, loss="least-squares",
    penalty="lasso", kappa_choice="ebic",
    optimizer="agd", lr_scheduler="fast-backtracking",
    max_iter=200, tol=1e-5,
    penalty_mu=False, gamma=1.0,
    verbose_bar=True, verbose=True,
    print_every=10, record_every=10)

learner_lasso_ebic.fit(X=data, end_time=T)
theta_hat_lasso_ebic = learner_lasso_ebic.estimated_params

print("Score learner lasso cv:", learner_lasso_cv.score(X=data, end_time=T))
print("Score learner lasso ebic:", learner_lasso_ebic.score(X=data, end_time=T))

# Plot the support of estimated coefficients ----------------------------------

learner_lasso_cv.plot_estimated_support()

learner_lasso_ebic.plot_estimated_support()
