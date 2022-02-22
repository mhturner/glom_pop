import os
import array
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.io import imsave

from glom_pop import dataio

raw_images_dir = os.path.join(dataio.get_config_file()['base_dir'], 'images', 'vh_iml')
output_images_dir = os.path.join(dataio.get_config_file()['base_dir'], 'images', 'vh_tif')
raw_images_dir
input_image_list = glob.glob(os.path.join(raw_images_dir, '*.iml'))
fh, ax = plt.subplots(4, 5, figsize=(15, 6))
ax = ax.ravel()
[x.set_axis_off() for x in ax]
im_names = []
for img_ind, image_path in enumerate(input_image_list):
    image_name = os.path.split(image_path)[-1].split('.')[0]
    with open(image_path, 'rb') as handle:
        s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(1024, 1536)
    img = np.uint8(255*(img / np.max(img)))
    # Trim to middle 512 (height) and all of 1536 (width)
    # Flip ud so [0,0] is on the ground when used as a texture in flystim
    img = np.flipud(img[256:768, :])

    ax[img_ind].imshow(img, origin='lower', cmap='Greys_r')
    im_names.append(image_name + '.tif')

    imsave(os.path.join(output_images_dir, image_name + '.tif'), img)

print(np.sort(im_names))

im_names
