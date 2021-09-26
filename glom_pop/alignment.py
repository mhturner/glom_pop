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
    """
    Remove gloms in mask with fewer than <threshold> number of voxels.

    :mask: ndarray of glom mask
    :threshold: minimum no. voxels for each included glom
    """
    vals = np.unique(mask)
    # mask out gloms with fewer than glom_size_threshold voxels
    for m_ind, mask_id in enumerate(vals):
        voxels_in_mask = np.sum(mask == mask_id)
        if voxels_in_mask < threshold:
            mask[mask == mask_id] = 0
        else:
            pass

    return mask


def filterGlomMask_by_name(mask, vpn_types, included_gloms):
    """
    Remove gloms in mask not in included_gloms list of names.

    :mask: ndarray of glom mask
    :vpn_types: DataFrame with val / glom name pairings
    :included_gloms: list of included glom names
    """
    included_vals = vpn_types.loc[vpn_types.get('vpn_types').isin(included_gloms), 'Unnamed: 0'].values
    all_vals = np.unique(mask)
    for m_ind, mask_id in enumerate(all_vals):
        if mask_id in included_vals:
            pass
        else:
            mask[mask == mask_id] = 0

    return mask
