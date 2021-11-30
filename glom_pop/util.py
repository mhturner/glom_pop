"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import matplotlib.pyplot as plt



def cleanAxes(ax):
    # ax.set_axis_off()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
