"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
import numpy as np


def getGlomResponses(brain, glom_mask, mask_values=None):
    if mask_values is None:
        mask_values = np.unique(glom_mask)
    glom_responses = [np.mean(brain[glom_mask == label, :], axis=0) for label in mask_values]
    return np.vstack(glom_responses)  # glom ID x Time


def getGlomVoxelResponses(brain, glom_mask, mask_values=None):
    if mask_values is None:
        mask_values = np.unique(glom_mask)
    glom_responses = [brain[glom_mask == label, :] for label in mask_values]
    return glom_responses  # list of len=gloms, each with array nvoxels x time


def filterGlomMask(mask, threshold):
    vals = np.unique(mask)
    # mask out gloms with fewer than glom_size_threshold voxels
    for m_ind, mask_id in enumerate(vals):
        voxels_in_mask = np.sum(mask == mask_id)
        if voxels_in_mask < threshold:
            mask[mask==mask_id] = 0
        else:
             pass

    return mask
