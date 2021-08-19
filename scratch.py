# %%

import pandas as pd
from scipy.io import mmread
import os

data_dir = '/Users/mhturner/Downloads'


# %%
a = mmread(os.path.join(data_dir, 'GSE156455_matrix_main.mtx.gz'))
a.shape
# %%
import scprep
data = scprep.io.load_10X(data_dir)

data = scprep.filter.remove_empty_cells(data)
data = scprep.filter.remove_empty_genes(data)

# %%



# %% Cluster voxel responses, with glom membership color code
import umap

map_vals = list(response_data['voxel_epoch_responses'].keys())

all_glom_ids = []
all_voxel_responses = []
for mv in map_vals:
    erm = response_data['voxel_epoch_responses'][mv]
    # Align responses
    mean_voxel_response, _, _, _, _, _ = ID.getMeanBrainByStimulus(erm)
    n_stimuli = mean_voxel_response.shape[2]
    concatenated_tuning = np.concatenate([mean_voxel_response[:, :, x] for x in range(n_stimuli)], axis=1)  # responses, time (concat stims)

    all_voxel_responses.append(concatenated_tuning)
    all_glom_ids.append(mv*np.ones(concatenated_tuning.shape[0]))

all_voxel_responses = np.vstack(all_voxel_responses)
all_glom_ids = np.hstack(all_glom_ids)

reducer = umap.UMAP()
embedding = reducer.fit_transform(all_voxel_responses)

# %%

cmap = cc.cm.glasbey
norm = mcolors.Normalize(vmin=0, vmax=vals.max(), clip=True)
fh, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(embedding[:, 0], embedding[:, 1], c=all_glom_ids, cmap=cmap, norm=norm, marker='.')
