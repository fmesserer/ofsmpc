###
# Create plot of the expected value of a normal distribution over a ReLU
###

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
from latexify import latexify, golden_ratio
from utils import expectation_over_relu

outfolder = 'plots_general/'
latexify(fig_width=3.5)

color_palette = sns.color_palette('viridis', as_cmap=True)
colors = [ color_palette(tau) for tau in np.linspace(0.4,.95,4) ]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

mu = np.linspace(-5,5,1000)  # mean
sigma = [.1, 1, 2, 5]        # standard deviation

plt.figure(figsize=(.8 * 3.5, .6 * golden_ratio * 3.5 ))

plt.plot(mu, np.maximum(0, mu), 'k:', label=r'$\max(0, \mu)$', zorder=10)   # ReLU
for s in sigma:
    plt.plot(mu, expectation_over_relu(mu, s), label=r'$\tilde\phi(\mu,{:.1f})$'.format(s))

plt.xlabel(r'mean $\mu$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim((-5,5))
plt.tight_layout(pad=0)
plt.savefig(outfolder + 'expectation_over_relu.pdf')
