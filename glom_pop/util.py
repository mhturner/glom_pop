"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
# from matplotlib.colors import LinearSegmentedColormap

#
# def getColors():
#     colors = [[0.000000, 0.317647, 0.000000],
#               [0.011765, 0.494118, 0.925490],
#               [0.000000, 0.843137, 0.000000],
#               [0.564706, 0.552941, 0.494118],
#               [0.031373, 0.827451, 0.878431],
#               [0.219608, 0.266667, 0.329412],
#               [1.000000, 0.937255, 0.345098],
#               [0.403922, 0.537255, 0.000000],
#               [0.000000, 0.509804, 0.517647],
#               [0.552941, 1.000000, 0.749020],
#               [0.262745, 0.701961, 0.501961],
#               [0.360784, 0.349020, 0.239216],
#               [0.709804, 0.745098, 0.803922],
#               [0.725490, 0.686275, 0.301961],
#               [0.423529, 0.705882, 1.000000],
#               [0.470588, 0.545098, 0.650980]]
#     colors = np.array(colors)
#     colors = np.concatenate((colors, np.ones(colors.shape[0])[:, np.newaxis]), axis=1)
#
#     return colors
#
#
# def getCmap():
#     cmap = LinearSegmentedColormap.from_list('cblind_glasbey', getColors(), 16)
#     return cmap


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


def makeGlomMap(ax, glom_map, z_val=None, highlight_vals=[]):
    vals = np.unique(glom_map)[1:]  # exclude first val (=0, not a glom)

    if highlight_vals == 'all':
        highlight_vals = vals

    if z_val is None:
        # Max projection across all Z
        slice = glom_map.max(axis=2)

        #  Special case: if single glom highlighted, make sure it shows up on "top" of the projection
        if len(highlight_vals) == 1:
            highlight_proj = np.any(glom_map == highlight_vals[0], axis=-1)
            slice[highlight_proj] = highlight_vals[0]
    else:
        # Specific Z value
        slice = glom_map[:, :, z_val]

    highlight_colors = cc.cm.glasbey(vals/vals.max())
    gray_color = cc.cm.gray(0.5)

    shape = list(slice.shape)
    shape.append(4)  # x, y, RGBA
    slice_rgb = np.zeros(shape)

    for v_ind, v in enumerate(vals):
        if v in highlight_vals:
            slice_rgb[slice == v, :] = highlight_colors[v_ind]
        else:
            slice_rgb[slice == v, :] = gray_color

    ax.imshow(np.swapaxes(slice_rgb, 0, 1), interpolation='none')
