"""
Align template brain & glom map to meanbrain

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import os
import numpy as np
import nibabel as nib
import pandas as pd
import colorcet as cc
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import ants
from matplotlib.patches import Rectangle
import glob

from glom_pop import alignment

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210824')
mask_fn = 'lobe_mask_chat_meanbrain_{}.nii'.format('20210824')

save_directory = os.path.join(base_dir, 'figs')
transform_directory = os.path.join(base_dir, 'transforms', 'meanbrain_template')


# %% Load

# Load meanbrain
meanbrain = ants.image_read(os.path.join(base_dir, 'mean_brain', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'mean_brain', mask_fn)).dataobj).astype('uint32')

# load transformed atlas and mask
glom_mask_2_meanbrain = ants.image_read(os.path.join(transform_directory, 'glom_mask_reg2meanbrain.nii')).numpy()
template_2_meanbrain = ants.image_read(os.path.join(transform_directory, 'JRC2018_reg2meanbrain.nii'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))


# %%
# Show z slices of meanbrain, template, & glom map for alignment
z_levels = [10, 20, 30, 40, 44]

glom_size_threshold = 350

glom_mask_2_meanbrain = alignment.filterGlomMask(glom_mask_2_meanbrain, glom_size_threshold)
vals = np.unique(glom_mask_2_meanbrain)[1:]  # exclude first val (=0, not a glom)
names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']


cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(glom_mask_2_meanbrain == 0, glom_mask_2_meanbrain)  # mask at 0
grn_tmp = np.ma.masked_where(lobe_mask == 0, meanbrain_green.numpy())  # mask at 0


fh, ax = plt.subplots(len(z_levels), 4, figsize=(8, 8))
[x.set_xticklabels([]) for x in ax.ravel()]
[x.set_yticklabels([]) for x in ax.ravel()]
[x.tick_params(bottom=False, left=False) for x in ax.ravel()]

vmax = np.quantile(grn_tmp[30:230, 5:180, :], 0.97)

for z_ind, z in enumerate(z_levels):
    # Note zoomed-in, trimmed view
    ax[z_ind, 0].imshow(meanbrain_red[30:230, 5:180, z].T, cmap='Reds')
    ax[z_ind, 1].imshow(grn_tmp[30:230, 5:180, z].T, cmap='Greens', vmax=vmax)
    ax[z_ind, 2].imshow(template_2_meanbrain[30:230, 5:180, z].T, cmap='Blues')
    ax[z_ind, 3].imshow(glom_tmp[30:230, 5:180, z].T, cmap=cmap, norm=norm, interpolation='none')

    if z_ind==0:
        ax[z_ind, 0].set_title('mtdTomato')
        ax[z_ind, 1].set_title('syt1GCaMP6F')
        ax[z_ind, 2].set_title('JRC2018')
        ax[z_ind, 3].set_title('Glomerulus map')

        dx = 25 / meanbrain_red.spacing[0]  # um -> pix
        ax[z_ind, 0].plot([10, 10+dx], [160, 160], color='k', linestyle='-', marker='None', linewidth=2)
        ax[z_ind, 0].annotate('25 um', (6, 150), fontweight='bold')

for x in ax.ravel():
    d_line = 30
    x.set_xticks(np.arange(d_line, 175, d_line))
    x.set_yticks(np.arange(d_line, 140, d_line))
    x.grid(which='major', axis='both', linestyle='--', color='k', linewidth=0.5)


handles = [Patch(facecolor=color) for color in colors]
fh.legend(handles, [label for label in names], fontsize=10, ncol=4, handleheight=1.0, labelspacing=0.05)

fh.savefig(os.path.join(save_directory, 'meanbrain_overlay.pdf'.format()))

# %%
cmap = 'binary_r'

dx = 100
dy = 60

box1_xy = (55, 10)
box2_xy = (30, 110)
box3_xy = (120, 5)

red_meanbrain = ants.split_channels(meanbrain)[0]
green_meanbrain = ants.split_channels(meanbrain)[1]

fh, ax = plt.subplots(1, 2, figsize=(6, 2))
[x.set_axis_off() for x in ax]
ax[0].imshow(green_meanbrain[:, :, 5:15].mean(axis=2).T, cmap=cmap, vmax=np.quantile(green_meanbrain.numpy(), 0.95))
rect1 = Rectangle(box1_xy, dx, dy, linewidth=2, edgecolor='m', facecolor='none')
ax[0].add_patch(rect1)

ax[1].imshow(green_meanbrain[:, :, 35:].mean(axis=2).T, cmap=cmap, vmax=np.quantile(green_meanbrain.numpy(), 0.95))
rect2 = Rectangle(box2_xy, dx, dy, linewidth=2, edgecolor='c', facecolor='none')
ax[1].add_patch(rect2)
rect3 = Rectangle(box3_xy, dx, dy, linewidth=2, edgecolor='b', facecolor='none')
ax[1].add_patch(rect3)

# %%
brain_directory = os.path.join(base_dir, 'anatomical_brains')
file_paths = glob.glob(os.path.join(brain_directory, '*_anatomical.nii'))

fh, ax = plt.subplots(3, len(file_paths)+1, figsize=(9, 2))

# [x.set_axis_off() for x in ax.ravel()]
ax[0, 0].imshow(green_meanbrain[box1_xy[0]:box1_xy[0]+dx, box1_xy[1]:box1_xy[1]+dy, 10].T, cmap=cmap)
ax[1, 0].imshow(green_meanbrain[box2_xy[0]:box2_xy[0]+dx, box2_xy[1]:box2_xy[1]+dy, 39].T, cmap=cmap)
ax[2, 0].imshow(green_meanbrain[box3_xy[0]:box3_xy[0]+dx, box3_xy[1]:box3_xy[1]+dy, 39].T, cmap=cmap)

ax[0, 0].imshow(glom_tmp[box1_xy[0]:box1_xy[0]+dx, box1_xy[1]:box1_xy[1]+dy, 10].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[1, 0].imshow(glom_tmp[box2_xy[0]:box2_xy[0]+dx, box2_xy[1]:box2_xy[1]+dy, 39].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')
ax[2, 0].imshow(glom_tmp[box3_xy[0]:box3_xy[0]+dx, box3_xy[1]:box3_xy[1]+dy, 39].T, cmap=cc.cm.glasbey, norm=norm, interpolation='none')

ax[0, 0].set_title('Mean')

[x.set_xticks([]) for x in ax.ravel()]
[x.set_yticks([]) for x in ax.ravel()]
[s.set_edgecolor('m') for s in ax[0, 0].spines.values()]
[s.set_edgecolor('c') for s in ax[1, 0].spines.values()]
[s.set_edgecolor('b') for s in ax[2, 0].spines.values()]

for f_ind, fp in enumerate(file_paths):
    series_name = os.path.split(fp)[-1].split('_')[0]
    disp = series_name.split('-')[1] + '-' + series_name.split('-')[2]
    base_dir = os.path.split(brain_directory)[0]
    transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_anatomical', series_name)
    brain_fp = os.path.join(transform_dir, 'meanbrain_reg.nii')
    ind_red = ants.split_channels(ants.image_read(brain_fp))[0]
    ind_green = ants.split_channels(ants.image_read(brain_fp))[0]

    ax[0, f_ind+1].imshow(ind_green[box1_xy[0]:box1_xy[0]+dx, box1_xy[1]:box1_xy[1]+dy, 10].T, cmap=cmap)
    ax[1, f_ind+1].imshow(ind_green[box2_xy[0]:box2_xy[0]+dx, box2_xy[1]:box2_xy[1]+dy, 39].T, cmap=cmap)
    ax[2, f_ind+1].imshow(ind_green[box3_xy[0]:box3_xy[0]+dx, box3_xy[1]:box3_xy[1]+dy, 39].T, cmap=cmap)

    ax[0, f_ind+1].set_title(disp, fontsize=6)

# %%
