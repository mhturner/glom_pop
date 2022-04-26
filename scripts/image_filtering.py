from flystim import image
import matplotlib.pyplot as plt
import numpy as np
from glom_pop import util
from skimage.io import imsave, imread
import os

# TODO: make glom_pop env work with flystim. Maybe update flystim pyqt to pyqt6 finally

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
    freq, pspect_orig = util.get_power_spectral_density(img_orig[:, 512:2*512], pixels_per_degree)
    original_spectra.append(pspect_orig)

    # High-pass
    filter_name = 'butterworth'
    filter_kwargs = {'cutoff_frequency_ratio': 0.1,
                     'order': 2,
                     'high_pass': True}

    img_hp = new_im.filter_image(filter_name=filter_name,
                                 filter_kwargs=filter_kwargs)
    freq, pspect_hp = util.get_power_spectral_density(img_hp[:, 512:2*512], pixels_per_degree)
    highpass_spectra.append(pspect_hp)

    # Low-pass
    filter_name = 'butterworth'
    filter_kwargs = {'cutoff_frequency_ratio': 0.1,
                     'order': 2,
                     'high_pass': False}

    img_lp = new_im.filter_image(filter_name=filter_name,
                                 filter_kwargs=filter_kwargs)
    freq, pspect_lp = util.get_power_spectral_density(img_lp[:, 512:2*512], pixels_per_degree)
    lowpass_spectra.append(pspect_lp)

    if False:
        img_white = new_im.whiten_image()
        imsave(os.path.join(save_dir, 'whitened_' + image_name), img_white)
    else:
        img_white = imread(os.path.join(save_dir, 'whitened_' + image_name))

    freq, pspect_white = util.get_power_spectral_density(img_white[:, 512:2*512], pixels_per_degree)
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

fh1, ax1 = plt.subplots(1, 1, figsize=(3.5, 3))
ax1.loglog(freq, original_spectra.mean(axis=0), label='Original')
ax1.loglog(freq, lowpass_spectra.mean(axis=0), label='Low pass')
ax1.loglog(freq, highpass_spectra.mean(axis=0), label='High pass')
ax1.loglog(freq, white_spectra.mean(axis=0), label='White')
ax1.set_xlabel('Freq. (cpd)')
ax1.set_ylabel('Power')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
fh1.legend()
ax1.set_xlim([1e-2, freq[-1]])

# %%
