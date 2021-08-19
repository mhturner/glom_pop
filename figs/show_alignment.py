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

from glom_pop import dataio, alignment

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
meanbrain_fn = 'chat_meanbrain_{}.nii'.format('20210816')
mask_fn = 'lobe_mask_chat_meanbrain_{}_ch1.nii'.format('20210816')

save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/figs'


# %% Load

# Load meanbrain
reference_fn = 'ZSeries-20210804-001'
filepath = os.path.join(base_dir, 'anatomical_brains', reference_fn)
metadata = dataio.get_bruker_metadata(filepath + '.xml')
meanbrain_red = dataio.get_ants_brain(os.path.join(base_dir, 'mean_brains', meanbrain_fn), metadata, channel=0)
meanbrain_green = dataio.get_ants_brain(os.path.join(base_dir, 'mean_brains', meanbrain_fn), metadata, channel=1)

lobe_mask = np.asanyarray(nib.load(os.path.join(base_dir, 'mean_brains', mask_fn)).dataobj).astype('uint32')

# load transformed atlas and mask
tmp_nib = nib.load(os.path.join(base_dir, 'aligned', 'glom_mask_reg2meanbrain.nii')).dataobj
glom_mask_2_meanbrain = np.squeeze(np.asanyarray(tmp_nib).astype('uint32'))

tmp_nib = nib.load(os.path.join(base_dir, 'aligned', 'JRC2018_reg2meanbrain.nii')).dataobj
template_2_meanbrain = np.squeeze(np.asanyarray(tmp_nib).astype('uint32'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# %%
# Show z slices of meanbrain, template, & glom map for alignment
z_levels = [10, 20, 30, 40]

glom_size_threshold = 350

glom_mask_2_meanbrain = alignment.filterGlomMask(glom_mask_2_meanbrain, glom_size_threshold)
vals = np.unique(glom_mask_2_meanbrain)[1:]  # exclude first val (=0, not a glom)
names = vpn_types.loc[vpn_types.get('Unnamed: 0').isin(vals), 'vpn_types']

cmap = cc.cm.glasbey
colors = cmap(vals/vals.max())
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
glom_tmp = np.ma.masked_where(glom_mask_2_meanbrain==0, glom_mask_2_meanbrain)  # mask at 0
grn_tmp = np.ma.masked_where(lobe_mask==0, meanbrain_green.numpy()) # mask at 0


fh, ax = plt.subplots(len(z_levels), 4, figsize=(8, 8))
[x.set_xticklabels([]) for x in ax.ravel()]
[x.set_yticklabels([]) for x in ax.ravel()]
[x.tick_params(bottom=False, left=False) for x in ax.ravel()]

for z_ind, z in enumerate(z_levels):
    # Note zoomed-in, trimmed view
    ax[z_ind, 0].imshow(meanbrain_red[55:230, 35:175, z].T, cmap='Reds')
    ax[z_ind, 1].imshow(grn_tmp[55:230, 35:175, z].T, cmap='Greens')
    ax[z_ind, 2].imshow(template_2_meanbrain[55:230, 35:175, z].T, cmap='Blues')
    ax[z_ind, 3].imshow(glom_tmp[55:230, 35:175, z].T, cmap=cmap, norm=norm, interpolation='none')

    if z_ind==0:
        ax[z_ind, 0].set_title('mtdTomato')
        ax[z_ind, 1].set_title('syt1GCaMP6F')
        ax[z_ind, 2].set_title('JRC2018')
        ax[z_ind, 3].set_title('Glomerulus map')

        dx = 25 / meanbrain_red.spacing[0]  # um -> pix
        ax[z_ind, 0].plot([10, 10+dx], [135, 135], color='k', linestyle='-', marker='None', linewidth=2)
        ax[z_ind, 0].annotate('25 um', (6, 130), fontweight='bold')

for x in ax.ravel():
    # x.locator_params(axis='y', nbins=6)
    # x.locator_params(axis='x', nbins=6)
    d_line = 30
    x.set_xticks(np.arange(d_line, 175, d_line))
    x.set_yticks(np.arange(d_line, 140, d_line))
    x.grid(which='major', axis='both', linestyle='--', color='k', linewidth=0.5)


handles = [Patch(facecolor=color) for color in colors]
fh.legend(handles, [label for label in names], fontsize=10, ncol=4, handleheight=1.0, labelspacing=0.05)

fh.savefig(os.path.join(save_directory, 'meanbrain_overlay.pdf'.format()))
