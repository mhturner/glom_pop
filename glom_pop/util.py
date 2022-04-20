"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from scipy import ndimage
import seaborn as sns
import pandas as pd

from glom_pop import dataio, alignment


class Tee:
    """
    Used to overwrite sys.stdout to print to sys.stdout as well as a txt file

    ref: https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
    """
    def write(self, *args, **kwargs):
        self.out1.write(*args, **kwargs)
        self.out2.write(*args, **kwargs)

    def __init__(self, out1, out2):
        self.out1 = out1
        self.out2 = out2

    def flush(self):
        pass


def getXcorr(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'same')
    return c


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


def get_color_dict():
    color_dict = {
                  'LC11': plt.get_cmap('tab20b')(0/20),
                  'LC21': plt.get_cmap('tab20b')(1/20),
                  'LC18': plt.get_cmap('tab20b')(2/20),

                  'LC6': plt.get_cmap('tab20b')(4/20),
                  'LC26': plt.get_cmap('tab20b')(5/20),
                  'LC16': plt.get_cmap('tab20b')(6/20),
                  'LPLC2': plt.get_cmap('tab20b')(7/20),

                  'LC4': plt.get_cmap('tab20b')(8/20),
                  'LPLC1': plt.get_cmap('tab20b')(9/20),
                  'LC9': plt.get_cmap('tab20b')(10/20),

                  'LC15': plt.get_cmap('tab20b')(16/20),
                  'LC12': plt.get_cmap('tab20b')(17/20),
                  'LC17': plt.get_cmap('tab20b')(18/20),

                  }

    return color_dict


def make_glom_map(ax, glom_map, z_val=None, highlight_names=[], colors='default'):
    # base_dir = dataio.get_config_file()['base_dir']
    # vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

    all_vals = np.unique(glom_map)[1:]  # exclude first val (=0, not a glom)
    # all_names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(all_vals), 'vpn_types']

    if highlight_names == 'all':
        highlight_vals = all_vals
    else:
        highlight_vals = dataio.get_glom_vals_from_names(highlight_names)

    if z_val is None:
        # Max projection across all Z
        slice = glom_map.max(axis=2)

        #  Special case: if single glom highlighted, make sure it shows up on "top" of the projection
        if len(highlight_names) == 1:
            highlight_proj = np.any(glom_map == highlight_vals[0], axis=-1)
            slice[highlight_proj] = highlight_vals[0]
    else:
        # Specific Z value
        slice = glom_map[:, :, z_val]

    gray_color = cc.cm.gray(0.75)

    shape = list(slice.shape)
    shape.append(4)  # x, y, RGBA
    slice_rgb = np.zeros(shape)  # Default: alpha=0

    for v_ind, val in enumerate(all_vals):
        if val in highlight_vals:
            if colors == 'default':
                highlight_color = list(get_color_dict()[dataio.get_glom_name_from_val(val)])  # Append alpha=1
            elif colors == 'glasbey':
                highlight_color = cc.cm.glasbey(all_vals/all_vals.max())[v_ind, :]

            slice_rgb[slice == val, :] = highlight_color
        else:
            slice_rgb[slice == val, :] = list(gray_color)

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
