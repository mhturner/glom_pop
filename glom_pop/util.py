"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc


def config_matplotlib():
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Helvetica'})


def clean_axes(ax):
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def make_glom_map(ax, glom_map, z_val=None, highlight_vals=[]):
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
