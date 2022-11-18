###
# based on https://nipunbatra.github.io/blog/2014/latexify.html
###

from math import sqrt
import matplotlib

golden_ratio = (sqrt(5)-1.0)/2.0    # Aesthetic ratio

def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        fig_height = fig_width*golden_ratio # height in inches

    params = {'backend': 'ps',
              'text.latex.preamble': [r"\usepackage{amsmath}"],
              'axes.labelsize': 8, # fontsize for x and y labels
              'axes.titlesize': 8,
              'legend.fontsize': 8, 
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'lines.markersize': 6 * fig_width / 5,        # scale relative to default figwidth
              'lines.linewidth': 1.5 * fig_width / 5,       # scale relative to default figwidth
    }

    matplotlib.rcParams.update(params)
    
