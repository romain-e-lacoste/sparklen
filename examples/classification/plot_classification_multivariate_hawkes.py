"""
===============================================================================
Classification of a bold_multivariate Hawkes Process (MHP)
===============================================================================

The following provides a comprehensive guide to using
:class:`~sparklen.hawkes.classification.ERMCLassifier` and
:class:`~sparklen.hawkes.classification.ERMLRCLassifier` for performing
classification of a MHP.
"""

# Author: Romain E. Lacoste
# License: BSD-3-Clause

# Setup environment -----------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split

from sparklen.hawkes.classification import (
    ERMCLassifier,
    ERMLRCLassifier,
    make_classification,
)
from sparklen.hawkes.inference import LearnerHawkesExp
from sparklen.hawkes.simulation import SimuHawkesExp
from sparklen.plot import plot_values

# Load the true coefficients --------------------------------------------------

K = 3
d = 15
beta = 3.0

bold_mu = np.empty((K,d))
bold_mu[0] = np.ones(d)*0.5
bold_mu[1] = np.ones(d)*0.5
bold_mu[2] = np.ones(d)*0.5

bold_alpha = np.empty((K,d,d))
bold_alpha[0] = np.zeros((d,d))
bold_alpha[1] = np.zeros((d,d))
bold_alpha[2] = np.zeros((d,d))

bold_alpha[0][:3, :3] += 0.2
bold_alpha[0][2:4, 2:4] += 0.2
bold_alpha[0][4:6, 4:6] += 0.2
bold_alpha[0][6:9, 6:9] += 0.2
bold_alpha[0][8:10, 8:10] += 0.2
bold_alpha[0][10:13, 10:13] += 0.2
bold_alpha[0][12:15, 12:15] += 0.2

bold_alpha[1][:2, :2] += 0.2
bold_alpha[1][2:4, 2:4] += 0.2
bold_alpha[1][3:6, 3:6] += 0.2
bold_alpha[1][6:9, 6:9] += 0.2
bold_alpha[1][8:10, 8:10] += 0.2
bold_alpha[1][10:12, 10:12] += 0.2
bold_alpha[1][12:15, 12:15] += 0.2

bold_alpha[2][:3, :3] += 0.2
bold_alpha[2][2:5, 2:5] += 0.2
bold_alpha[2][5:8, 5:8] += 0.2
bold_alpha[2][7:9, 7:9] += 0.2
bold_alpha[2][9:11, 9:11] += 0.2
bold_alpha[2][11:13, 11:13] += 0.2
bold_alpha[2][12:15, 12:15] += 0.2

bold_theta_star = np.empty((K, d, d + 1))
for k in range(K):
    bold_theta_star[k] = np.hstack([np.reshape(
        bold_mu[k], (d, -1)),
        bold_alpha[k]])

# Plot the true coefficients --------------------------------------------------

plot_values(bold_theta_star[1])

# Simulate training data ------------------------------------------------------

T = 5.0
n = 600

X, y = make_classification(
    bold_mu=bold_mu, bold_alpha=bold_alpha,
    beta=beta, end_time=T,
    n_samples=n, n_classes=K,
    random_state=4)

# Split the sample into training and test samples.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=4)

# Perform classification ------------------------------------------------------

clf_erm = ERMCLassifier(
    decay=beta, gamma0=0.1,
    max_iter=500, tol=1e-6,
    verbose_bar=True, verbose=True,
    print_every=10)

clf_erm.fit(X=X_train, y=y_train, end_time=T)

clf_ermlr = ERMLRCLassifier(
    decay=beta, gamma0=0.1,
    max_iter=500, tol=1e-6,
    verbose_bar=True, verbose=True,
    print_every=10)

clf_ermlr.fit(X=X_train, y=y_train, end_time=T)

# Print the score on test sample ----------------------------------------------

print(clf_erm.score(X=X_test, y=y_test, end_time=T))

print(clf_ermlr.score(X=X_test, y=y_test, end_time=T))

# Plot the confusion matrix ---------------------------------------------------

clf_erm.plot_score_cm(X=X_test, y=y_test, end_time=T)

clf_ermlr.plot_score_cm(X=X_test, y=y_test, end_time=T)
