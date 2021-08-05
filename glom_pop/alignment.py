"""

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""


def getGlomResponses(brain, glom_mask, mask_values=None):
    if mask_values is None:
        mask_values = np.unique(glom_mask)
    glom_responses = [np.mean(brain[glom_mask==label, :], axis=0) for label in mask_values]
    return np.vstack(glom_responses) # glom ID x Time
