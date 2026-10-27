"""
===============================================================================
Simulation of a Multivariate Hawkes Process (MHP)
===============================================================================

The following provides a comprehensive guide to using
:class:`~sparklen.hawkes.simulation.SimuHawkesExp` to simulate the
events of a MHP.
"""

# Author: Romain E. Lacoste
# License: BSD-3-Clause

# Setup environment -----------------------------------------------------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sparklen.hawkes.simulation import SimuHawkesExp
from sparklen.plot import plot_values

# Set the true coefficients ---------------------------------------------------

d = 2
beta = 3.0

mu = np.array([0.6, 0.5])
alpha = np.array([[0.2, 0.1], [0.0, 0.3]])

theta_star = np.hstack([np.reshape(mu, (d,-1)), alpha])

# Plot the true coefficients --------------------------------------------------

plot_values(theta_star)

# Simulate data ---------------------------------------------------------------

T = 5.0
n = 3

hawkes = SimuHawkesExp(
    mu=mu, alpha=alpha, beta=beta,
    end_time=T, n_samples=n,
    random_state=8)

hawkes.simulate()

data = hawkes.timestamps
print(data)

# Plot the repetitions --------------------------------------------------------

# Number of dimensions
num_dimensions = len(data[0])

# Create a figure with two subplots (stacked vertically)
fig, axes = plt.subplots(num_dimensions, 1, figsize=(6, 5), sharex=True)

# Gradient of blue color map
num_repetitions = len(data)
colors = cm.Blues(np.linspace(0.3, 0.9, num_repetitions))

# Loop over dimensions
for dim in range(num_dimensions):
    ax = axes[dim] if num_dimensions > 1 else axes

    # Loop over repetitions
    for rep_idx, repetition in enumerate(data):
        jump_times = repetition[dim]
        num_jumps = len(jump_times)
        color = colors[rep_idx]

        # Plotting for each repetition
        ax.plot(0, 0, 'o', label=f'Repetition {rep_idx+1}', color=color)
        if num_jumps > 0:
            ax.hlines(y=0, xmin=0, xmax=jump_times[0], color=color)
            for i in range(1, num_jumps):
                ax.plot(jump_times[i-1], i, 'o', color=color)
                ax.hlines(y=i, xmin=jump_times[i-1],
                          xmax=jump_times[i], color=color)
                ax.vlines(x=jump_times[i-1], ymin=i-1,
                          ymax=i, linestyle=':', color=color)

            # Last point
            ax.plot(jump_times[-1], num_jumps, 'o', color=color)
            ax.hlines(y=num_jumps, xmin=jump_times[-1],
                      xmax=T, color=color)
            ax.vlines(x=jump_times[-1], ymin=num_jumps-1,
                      ymax=num_jumps, linestyle=':', color=color)

        ax.plot(T, num_jumps, 'x', color=color)
        ax.vlines(x=T, ymin=0, ymax=num_jumps,
                  linestyle=':', color='#C44E52')

    # Add labels and title
    ax.set_ylabel(f'$N_{dim+1}(t)$', fontsize=12)
    if dim == num_repetitions-2:
        ax.set_xlabel('Jump times', fontsize=12)
    ax.set_xlim(-0.1, T + 0.1)
    ax.legend()

# Adjust layout
fig.tight_layout()
