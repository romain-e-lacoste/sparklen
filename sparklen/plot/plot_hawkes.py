# Author: Romain E. Lacoste
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors, gridspec
from sklearn.metrics import confusion_matrix

from sparklen.plot.plot_utils import get_color, setup_latex_plotting


def _get_tick_frequence(d):
    """
    Determine the y-tick interval based on the value of d.
    """

    if d <= 5:
        return 1  # Show all ticks
    elif 5 < d <= 15:
        return 2  # Show every 2nd tick
    elif 15 < d <= 25:
        return 3  # Show every 3nd tick
    else:
        return d // 5  # Show a maximum of 5 ticks


def _get_fig_size(d):
    """
    Returns a figure size that scales smoothly with d using a polynomial function.
    """

    # Use a polynomial function for smooth growth
    a = 1.0  # Base size
    b = 0.4  # Degree of growth

    return a + d ** b

def plot_values(array, save_path=None, save_format='png', dpi=300, use_latex=False):
    """
    Plot values of a 2D array as a heatmap.

    The first column and the remaining matrix values are displayed,
    each with its own colorbar for better visualization.

    Parameters
    ----------
    array : ndarray of shape (d, d+1)
        The array to be plotted

    save_path : str, optional, default=None
        The path where the plot will be saved. If not provided, the plot will not be saved.

    save_format : str, optional, default='png'
        The format in which to save the plot (e.g., 'png', 'pdf', 'pgf').

    dpi : int, optional, default=300
        The resolution of the saved plot. Higher values result in higher quality.

    use_latex : bool, optional, default=False
        Whether to use LaTeX for rendering text in the plot.
        If `True`, text will be rendered using LaTeX formatting.
    """
    # Apply LaTeX styling if use_latex=True
    if use_latex:
        setup_latex_plotting()

    # Ensure the array has the shape (d, d+1)
    d, d_plus_one = array.shape
    assert d_plus_one == d + 1, "Array must have shape (d, d+1)"

    frequence = _get_tick_frequence(d)

    # Separate the first column and the matrix
    mu = array[:, [0]]  # Shape (d, 1)
    alpha = array[:, 1:]          # Shape (d, d)

    # Set the figure size based on d
    alpha_size = _get_fig_size(d)  # Base width for the heatmap
    fig = plt.figure(figsize=(alpha_size + 3, alpha_size))  # Extra width for colorbars

    # Create a gridspec layout
    spec = gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[0.5, alpha_size, 0.25, 0.25], figure=fig)

    # Plot the mu heatmap
    ax1 = fig.add_subplot(spec[0, 0])
    sns.heatmap(mu, ax=ax1, cmap='Reds', cbar=False, xticklabels=False, yticklabels=frequence, vmin=0.0)
    ax1.set_title(r'$\mu$', pad=10)

    # Plot the alpha heatmap
    ax2 = fig.add_subplot(spec[0, 1])
    sns.heatmap(alpha, ax=ax2, cmap='Blues', cbar=False, xticklabels=frequence, yticklabels=frequence, vmin=0.0)
    ax2.set_title(r'$\alpha$', pad=10)
    ax2.set_aspect('equal') # Square matrix rendering

    # Add the mu colorbar
    cax1 = fig.add_subplot(spec[0, 2])
    sns.heatmap(mu, ax=ax1, cmap='Reds', cbar=True, cbar_ax=cax1, cbar_kws={'orientation': 'vertical'}, xticklabels=False, yticklabels=frequence, vmin=0.0)
    ax1.set_yticklabels([str(i) for i in range(1, d+1, frequence)], rotation=0)

    # Add the alpha colorbar
    cax2 = fig.add_subplot(spec[0, 3])
    sns.heatmap(alpha, ax=ax2, cmap='Blues', cbar=True, cbar_ax=cax2, cbar_kws={'orientation': 'vertical'}, xticklabels=frequence, yticklabels=frequence, vmin=0.0)
    ax2.set_xticklabels([str(i) for i in range(1, d+1, frequence)], rotation=0)
    ax2.set_yticklabels([str(i) for i in range(1, d+1, frequence)], rotation=0)

    # Adjust the layout to ensure all elements fit well
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(f"{save_path}.{save_format}", format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {save_path}.{save_format}")

    # Show the plot
    plt.show()

def plot_support(array, save_path=None, save_format='png', dpi=300, use_latex=False):
    """
    Plot support of a 2D array as a heatmap.

    The first column and the remaining matrix values are displayed,
    with zero and non-zero values highlighted in different colors.

    Parameters
    ----------
    array : ndarray of shape (d, d+1)
        The array to be plotted

    save_path : str, optional, default=None
        The path where the plot will be saved. If not provided, the plot will not be saved.

    save_format : str, optional, default='png'
        The format in which to save the plot (e.g., 'png', 'pdf', 'pgf').

    dpi : int, optional, default=300
        The resolution of the saved plot. Higher values result in higher quality.

    use_latex : bool, optional, default=False
        Whether to use LaTeX for rendering text in the plot.
        If `True`, text will be rendered using LaTeX formatting.
    """
    # Apply LaTeX styling if use_latex=True
    if use_latex:
        setup_latex_plotting()

    # Ensure the array has the shape (d, d+1)
    d, d_plus_one = array.shape
    assert d_plus_one == d + 1, "Array must have shape (d, d+1)"

    frequence = _get_tick_frequence(d)

    # Convert array to binary support (0 for 0, 1 for any positive value)
    support = (array > 0).astype(int)
    mu_support = support[:, [0]]  # Shape (d, 1)
    alpha_support = support[:, 1:]         # Shape (d, d)

    # Define figure size based on d
    fig_size = _get_fig_size(d)
    fig = plt.figure(figsize=(fig_size + 2, fig_size))

    # Create a GridSpec layout with three columns
    spec = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[0.5, fig_size, 0.25], figure=fig)

    # Create custom binary colormap (e.g., light grey for 0, green for 1)
    cmap = colors.ListedColormap([get_color('deep_blue'), get_color('bright_yellow')])

    # Parameter mu support heatmap
    ax1 = fig.add_subplot(spec[0, 0])
    sns.heatmap(mu_support, ax=ax1, cmap=cmap, cbar=False, xticklabels=False, yticklabels=frequence, vmin=0, vmax=1)
    ax1.set_yticklabels([str(i) for i in range(1, d+1, frequence)], rotation=0)
    ax1.set_title(r'$\mu$', pad=10)

    # Parameter alpha support heatmap
    ax2 = fig.add_subplot(spec[0, 1])
    sns.heatmap(alpha_support, ax=ax2, cmap=cmap, cbar=False, xticklabels=frequence, yticklabels=frequence)
    ax2.set_title(r'$\alpha$', pad=10)
    ax2.set_aspect('equal') # Square matrix rendering

    # Colorbar for both heatmaps
    axcb = fig.add_subplot(spec[0, 2])
    sns.heatmap(alpha_support, ax=ax2, cmap=cmap, cbar=True, cbar_ax=axcb, xticklabels=frequence, yticklabels=frequence, cbar_kws={'ticks': [0.25, 0.75]})
    axcb.set_yticklabels(['0', '1'])  # Single label for binary colorbar
    ax2.set_xticklabels([str(i) for i in range(1, d+1, frequence)], rotation=0)
    ax2.set_yticklabels([str(i) for i in range(1, d+1, frequence)], rotation=0)

    # Adjust the layout to ensure all elements fit well
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(f"{save_path}.{save_format}", format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {save_path}.{save_format}")

    # Show the plot
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None, save_format='png', dpi=300, use_latex=False):
    """
    Plot a confusion matrix to visualize classification performance.

    The confusion matrix compares true labels with predicted labels,
    showing counts of correct and incorrect predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true class labels.

    y_pred : array-like of shape (n_samples,)
        The predicted class labels.

    save_path : str, optional, default=None
        The path where the plot will be saved. If not provided, the plot will not be saved.

    save_format : str, optional, default='png'
        The format in which to save the plot (e.g., 'png', 'pdf', 'pgf').

    dpi : int, optional, default=300
        The resolution of the saved plot. Higher values result in higher quality.

    use_latex : bool, optional, default=False
        Whether to use LaTeX for rendering text in the plot.
        If `True`, text will be rendered using LaTeX formatting.
    """
    # Apply LaTeX styling if use_latex=True
    if use_latex:
        setup_latex_plotting()

    class_labels = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    #fig_size = _get_fig_size(class_labels.shape[0])

    fig, (ax, axcb) = plt.subplots(1, 2, figsize=(5, 4), sharex=False, sharey=False, gridspec_kw={'width_ratios': [4, 0.25]})
    sns.heatmap(cm, square=True, annot=True, cmap='Blues', ax=ax, cbar_ax=axcb, vmin=0, vmax=1, xticklabels=class_labels, yticklabels=class_labels)
    plt.yticks(rotation=0)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_ylim(len(cm), 0)

    # Adjust the layout to ensure all elements fit well
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(f"{save_path}.{save_format}", format=save_format, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {save_path}.{save_format}")

    # Show the plot
    plt.show()
