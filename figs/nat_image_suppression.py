from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
included_vals = dataio.get_glom_vals_from_names(included_gloms)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'NaturalImageSuppression',
                                                                          'include_in_analysis': True,
                                                                          })

# %%

# shape = gloms, images, filter conditions, flies, time
all_responses = np.zeros( (len(included_gloms), 5, 2, len(matching_series), 43) )
all_responses[:] = np.nan
all_responses.shape
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Align responses
    parameter_key = ('stim0_image_name', 'current_filter_flag')
    unique_parameter_values, mean_response, sem_response, trial_response_by_stimulus = ID.getTrialAverages(epoch_response_matrix, parameter_key=parameter_key)
    image_names = np.unique([x.replace('whitened_', '') for x in unique_parameter_values[:, 0]])
    filter_codes = [0, 1]  # 0=raw, 1=whitened, 2=DoG
    for f_ind, filter_code in enumerate(filter_codes):
        for im_ind, image_name in enumerate(image_names):
            pull_ind = np.where(np.logical_and([image_name in x for x in unique_parameter_values[:, 0]],
                                               [filter_code == float(x) for x in unique_parameter_values[:, 1]]))[0][0]
            all_responses[:, im_ind, f_ind, s_ind, :] = mean_response[:, pull_ind, :]

# %%
all_responses.shape
image_names
fh, ax = plt.subplots(len(included_gloms), 5, figsize=(4, 8))
[x.set_ylim([-0.1, 0.35]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()]
for g_ind, glom in enumerate(included_gloms):
    for f_ind, filter_code in enumerate(filter_codes):
        if f_ind == 0:
            alpha = 1.0
        else:
            alpha = 0.5
        for im_ind, image_name in enumerate(image_names):
            ax[g_ind, im_ind].plot(all_responses[g_ind, im_ind, f_ind, :, :].mean(axis=0),
                                   color=util.get_color_dict()[glom],
                                   alpha=alpha)


# %%

all_amplitudes = ID.getResponseAmplitude(all_responses, metric='max')

fh, ax = plt.subplots(len(included_gloms), 1, figsize=(0.75, 8))
[x.set_xticks([]) for x in ax]
[x.set_yticks([]) for x in ax]
[x.set_xlim([0, 0.7]) for x in ax]
[x.set_ylim([0, 0.7]) for x in ax]
[x.spines['top'].set_visible(False) for x in ax]
[x.spines['right'].set_visible(False) for x in ax]
for g_ind, glom in enumerate(included_gloms):
    ax[g_ind].plot([0, 0.7], [0, 0.7], 'k-', alpha=0.5, zorder=0)
    ax[g_ind].scatter(all_amplitudes[g_ind, :, 0, :].mean(axis=0),
                      all_amplitudes[g_ind, :, 1, :].mean(axis=0),
                      color=util.get_color_dict()[glom])
    result = ttest_rel(all_amplitudes[g_ind, :, 0, :].mean(axis=0),
                       all_amplitudes[g_ind, :, 1, :].mean(axis=0))





# %%
