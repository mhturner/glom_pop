from flystim import image
import matplotlib.pyplot as plt
import numpy as np
from glom_pop import util

# TODO: make glom_pop env work with flystim. Maybe update flystim pyqt to pyqt6 finally

low_sigma = 1  # Deg.
high_sigma = 4  # Deg.
pixels_per_degree = 1536 / 360

image_names = ['imk00152.tif', 'imk00377.tif', 'imk00405.tif', 'imk00459.tif',
               'imk00657.tif', 'imk01151.tif', 'imk01154.tif', 'imk01192.tif',
               'imk01769.tif', 'imk01829.tif', 'imk02265.tif', 'imk02281.tif',
               'imk02733.tif', 'imk02999.tif', 'imk03093.tif', 'imk03347.tif',
               'imk03447.tif', 'imk03584.tif', 'imk03758.tif', 'imk03760.tif']

filter_name = 'difference_of_gaussians'
filter_kwargs = {'low_sigma': low_sigma * pixels_per_degree,  # degrees -> pixels
                 'high_sigma': high_sigma * pixels_per_degree}  # degrees -> pixels


eg_image_inds = [0, 3, 15]
fh0, ax0 = plt.subplots(len(eg_image_inds), 3, figsize=(8, 4), tight_layout=True)
[x.set_axis_off() for x in ax0.ravel()]

filtered_spectra = []
original_spectra = []
white_spectra = []
for im_ind, image_name in enumerate(image_names):
    new_im = image.Image(image_name=image_name)
    img_orig = new_im.load_image()
    freq, pspect_orig = util.get_power_spectral_density(img_orig[:, 512:2*512], pixels_per_degree)
    original_spectra.append(pspect_orig)

    img_filt = new_im.filter_image(filter_name=filter_name,
                                   filter_kwargs=filter_kwargs)
    freq, pspect_filt = util.get_power_spectral_density(img_filt[:, 512:2*512], pixels_per_degree)
    filtered_spectra.append(pspect_filt)

    img_white = new_im.whiten_image()
    freq, pspect_white = util.get_power_spectral_density(img_white[:, 512:2*512], pixels_per_degree)
    white_spectra.append(pspect_white)

    if im_ind in eg_image_inds:
        ax_ind = np.where(np.array(eg_image_inds) == im_ind)[0][0]
        ax0[ax_ind, 0].imshow(img_orig, cmap='Greys_r', origin='lower')
        ax0[ax_ind, 1].imshow(img_filt, cmap='Greys_r', origin='lower')
        ax0[ax_ind, 2].imshow(img_white, cmap='Greys_r', origin='lower')

original_spectra = np.vstack(original_spectra)
filtered_spectra = np.vstack(filtered_spectra)
white_spectra = np.vstack(white_spectra)

ax0[0, 0].set_title('Original')
ax0[0, 1].set_title('Filtered')
ax0[0, 2].set_title('Whitened')

# %%

fh1, ax1 = plt.subplots(1, 1, figsize=(3.5, 3))
ax1.loglog(freq, original_spectra.mean(axis=0), label='Original')
ax1.loglog(freq, filtered_spectra.mean(axis=0), label='Filtered')
ax1.loglog(freq, white_spectra.mean(axis=0), label='Whitened')
ax1.set_xlabel('Freq. (cpd)')
ax1.set_ylabel('Power')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
fh1.legend()




# %%
