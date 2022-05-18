from flystim import image
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread
import os
from scipy import ndimage

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

# %%

pixels_per_degree = 1536 / 360
screen_width = 160 * pixels_per_degree  # deg -> pixels
screen_height = 50 * pixels_per_degree  # deg -> pixels

image_names = ['imk00152.tif', 'imk00377.tif', 'imk00405.tif', 'imk00459.tif',
               'imk00657.tif', 'imk01151.tif', 'imk01154.tif', 'imk01192.tif',
               'imk01769.tif', 'imk01829.tif', 'imk02265.tif', 'imk02281.tif',
               'imk02733.tif', 'imk02999.tif', 'imk03093.tif', 'imk03347.tif',
               'imk03447.tif', 'imk03584.tif', 'imk03758.tif', 'imk03760.tif']

save_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/images/vh_whitened'

eg_image_inds = [0, 5, 15]
fh0, ax0 = plt.subplots(len(eg_image_inds), 4, figsize=(8, 4), tight_layout=True)
[x.set_axis_off() for x in ax0.ravel()]

original_spectra = []
white_spectra = []
highpass_spectra = []
lowpass_spectra = []
for im_ind, image_name in enumerate(image_names):
    new_im = image.Image(image_name=image_name)
    img_orig = new_im.load_image()
    freq, pspect_orig = get_power_spectral_density(img_orig[:, 512:2*512], pixels_per_degree)
    original_spectra.append(pspect_orig)

    # High-pass
    filter_name = 'butterworth'
    filter_kwargs = {'cutoff_frequency_ratio': 0.1,
                     'order': 2,
                     'high_pass': True}

    img_hp = new_im.filter_image(filter_name=filter_name,
                                 filter_kwargs=filter_kwargs)
    freq, pspect_hp = get_power_spectral_density(img_hp[:, 512:2*512], pixels_per_degree)
    highpass_spectra.append(pspect_hp)

    # Low-pass
    filter_name = 'butterworth'
    filter_kwargs = {'cutoff_frequency_ratio': 0.1,
                     'order': 2,
                     'high_pass': False}

    img_lp = new_im.filter_image(filter_name=filter_name,
                                 filter_kwargs=filter_kwargs)
    freq, pspect_lp = get_power_spectral_density(img_lp[:, 512:2*512], pixels_per_degree)
    lowpass_spectra.append(pspect_lp)

    if False:
        img_white = new_im.whiten_image()
        imsave(os.path.join(save_dir, 'whitened_' + image_name), img_white)
    else:
        img_white = imread(os.path.join(save_dir, 'whitened_' + image_name))

    freq, pspect_white = get_power_spectral_density(img_white[:, 512:2*512], pixels_per_degree)
    white_spectra.append(pspect_white)

    if im_ind in eg_image_inds:
        ax_ind = np.where(np.array(eg_image_inds) == im_ind)[0][0]
        ax0[ax_ind, 0].imshow(img_orig, cmap='Greys_r', origin='lower')
        ax0[ax_ind, 1].imshow(img_lp, cmap='Greys_r', origin='lower')
        ax0[ax_ind, 2].imshow(img_hp, cmap='Greys_r', origin='lower')
        ax0[ax_ind, 3].imshow(img_white, cmap='Greys_r', origin='lower')

original_spectra = np.vstack(original_spectra)
highpass_spectra = np.vstack(highpass_spectra)
lowpass_spectra = np.vstack(lowpass_spectra)
white_spectra = np.vstack(white_spectra)

ax0[0, 0].set_title('Original')
ax0[0, 1].set_title('Low pass')
ax0[0, 2].set_title('High pass')
ax0[0, 3].set_title('Whitened')

image_width = img_orig.shape[1]
image_height = img_orig.shape[0]

# Crop to about the extent of the image on the screen
[x.set_xlim([image_width/2 - screen_width/2, image_width/2 + screen_width/2]) for x in ax0.ravel()]
[x.set_ylim([image_height/2 - image_height/2, image_height/2 + image_height/2]) for x in ax0.ravel()];

# %%

fh1, ax1 = plt.subplots(1, 1, figsize=(3, 3))
ax1.loglog(freq, original_spectra.mean(axis=0), label='Original', color='k')
ax1.loglog(freq, white_spectra.mean(axis=0), label='Whitened', color='b')
ax1.loglog(freq, highpass_spectra.mean(axis=0), label='Highpass', color='m')
ax1.loglog(freq, lowpass_spectra.mean(axis=0), label='Lowpass', color='y')


ax1.set_xlabel('Freq. (cpd)')
ax1.set_ylabel('Power')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
fh1.legend()
ax1.set_xlim([1e-2, freq[-1]]);

fig_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'
fh1.savefig(os.path.join(fig_dir, 'nat_image_pspects.svg'), transparent=True)

# %%
