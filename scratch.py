import pandas as pd
import colorcet as cc
import pandas as pd
import os
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

dir = '/Users/mhturner/GitHub/glom_pop'

cmap = cc.cm.glasbey
colors = [[0.000000, 0.317647, 0.000000],
          [0.011765, 0.494118, 0.925490],
          [0.000000, 0.843137, 0.000000],
          [0.564706, 0.552941, 0.494118],
          [0.031373, 0.827451, 0.878431],
          [0.219608, 0.266667, 0.329412],
          [1.000000, 0.937255, 0.345098],
          [0.403922, 0.537255, 0.000000],
          [0.000000, 0.509804, 0.517647],
          [0.552941, 1.000000, 0.749020],
          [0.262745, 0.701961, 0.501961],
          [0.360784, 0.349020, 0.239216],
          [0.709804, 0.745098, 0.803922],
          [0.725490, 0.686275, 0.301961],
          [0.423529, 0.705882, 1.000000],
          [0.470588, 0.545098, 0.650980]]
colors = np.array(colors)
colors = np.concatenate((colors, np.ones(colors.shape[0])[:, np.newaxis]), axis=1)


cmap = LinearSegmentedColormap.from_list('cblind_glasbey', colors, 16)
cmap
