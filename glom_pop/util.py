"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from scipy import ndimage


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


def get_power_spectral_density(image_array, pixels_per_degree):
    """
    Return 1D power spectral density for an image.
    Params:
        :image_array: ndarray. Needs to be square.
        :pixels_per_degree: scale of image, to return frequency in cycles per degree

    Returns:
        :freq: 1DF array of frequency (cycles per degree)
        :psd1D: 1D array of power spectrum

    """
    assert image_array.shape[0] == image_array.shape[1], 'Input must be square image array'

    fft_2d = np.abs(np.fft.fftshift(np.fft.fft2(image_array[:, :512])))**2
    ndim = fft_2d.shape[0]

    # Circular sum to collapse into 1D power spectrum
    # Ref: https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    h = fft_2d.shape[0]
    w = fft_2d.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc).astype(int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.sum(fft_2d, r, index=np.arange(0, wc))

    freq = np.fft.fftfreq(ndim, d=pixels_per_degree)[:ndim//2]

    return freq, psd1D
