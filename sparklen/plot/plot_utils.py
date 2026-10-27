# Author: Romain E. Lacoste
# License: BSD-3-Clause

import matplotlib.pyplot as plt

# Named color dictionary
named_colors = {
    'deep_blue': '#0d0887',
    'bright_yellow': '#f0f921',
    # Add more named colors as needed
}

def setup_latex_plotting():
    """Configure Matplotlib for LaTeX rendering and set font parameters."""
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.titlesize'] = 18

def get_color(name):
    """Return a color by name from the named_colors dictionary."""
    return named_colors.get(name, '#000000')  # Default to black if name is not found
