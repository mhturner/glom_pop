from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from glom_pop import dataio, util
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)

eg_series = ('2022-03-18', 7)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'CoherentDots',
                                                                          'include_in_analysis': True,
                                                                          'signal_direction': 0.0,
                                                                          },
                                                  target_groups=['aligned_response'],
                                                  )

for series in np.array(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    print('{}: {}'.format(file_name, series_number))
# %%
target_coherence = [0, 0.25, 0.5, 0.75, 1.0]
target_speed = 80

all_responses = []
response_amplitudes = []
for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    ID = imaging_data.ImagingDataObject(file_path,
                                        series_number,
                                        quiet=True)

    # Load response data
    response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    # Select only trials with target params:
    # Shape = (gloms x param conditions x time)
    trial_averages = np.zeros((len(included_gloms), len(target_coherence), epoch_response_matrix.shape[-1]))
    trial_averages[:] = np.nan
    num_matching_trials = []
    for coh_ind, coh in enumerate(target_coherence):
        erm_selected = shared_analysis.filterTrials(epoch_response_matrix,
                                                    ID,
                                                    query={'current_coherence': coh,
                                                           'current_speed': target_speed},
                                                    return_inds=False)
        num_matching_trials.append(erm_selected.shape[1])
        trial_averages[:, coh_ind, :] = np.nanmean(erm_selected, axis=1)  # each trial average: gloms x time

    if np.all(np.array(num_matching_trials) > 0):
        print('Adding fly from {}: {}'.format(os.path.split(file_path)[-1], series_number))
        response_amp = ID.getResponseAmplitude(trial_averages, metric='max')  # shape = gloms x param condition

        all_responses.append(trial_averages)
        response_amplitudes.append(response_amp)

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        # eg fly: show responses to 0 and 1 coherence
        fh0, ax = plt.subplots(2, len(included_gloms), figsize=(4, 1.25), gridspec_kw={'hspace': 0})
        [x.set_ylim([-0.15, 1.0]) for x in ax.ravel()]
        [x.set_xlim([-0.25, response_data['time_vector'].max()]) for x in ax.ravel()]
        [util.clean_axes(x) for x in ax.ravel()]
        for g_ind, glom in enumerate(included_gloms):
            ax[0, g_ind].set_title(glom, fontsize=9, rotation=45)

            for u_ind, up in enumerate([0, 1]):
                pull_ind = np.where(np.array(target_coherence) == up)[0]
                if u_ind == 0:
                    plot_tools.addScaleBars(ax[0, 0], dT=2, dF=0.25, T_value=-0.1, F_value=-0.1)
                ax[u_ind, g_ind].axhline(y=0, color='k', alpha=0.5)
                ax[u_ind, g_ind].plot(response_data['time_vector'],
                                      trial_averages[g_ind, pull_ind, :][0],
                                      color=util.get_color_dict()[glom])


# Stack accumulated responses
# The glom order here is included_gloms
all_responses = np.stack(all_responses, axis=-1)  # dims = (glom, param, time, fly)
response_amplitudes = np.stack(response_amplitudes, axis=-1)  # dims = (gloms, param, fly)

# stats across animals
mean_responses = np.nanmean(all_responses, axis=-1)  # (glom, param, time)
sem_responses = np.nanstd(all_responses, axis=-1) / np.sqrt(all_responses.shape[-1])  # (glom, param, time)
std_responses = np.nanstd(all_responses, axis=-1)  # (glom, param, time)

fh0.savefig(os.path.join(save_directory, 'coherence_eg_fly.svg'), transparent=True)

# %% Normalized response amplitudes. Norm by peak response to any coherence condition

# resp amp normalized to 0 coherence, within fly
normalized_response_amplitudes = response_amplitudes / response_amplitudes[:, 0, :][:, np.newaxis, :]
fh2, ax = plt.subplots(1, len(included_gloms), figsize=(4, 1.25))
[x.set_ylim([0.3, 1.5]) for x in ax.ravel()]
[util.clean_axes(x) for x in ax.ravel()[1:]]
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

p_vals = []
for g_ind, glom in enumerate(included_gloms):
    # Ttest 0 vs. 1 coherence
    h, p = ttest_rel(normalized_response_amplitudes[g_ind, 0, :], normalized_response_amplitudes[g_ind, -1, :])
    p_vals.append(p)

    ax[g_ind].axhline(y=1, color='k', alpha=0.50)
    ax[g_ind].set_title(glom, fontsize=9)
    ax[g_ind].plot(target_coherence, normalized_response_amplitudes[g_ind, :, :].mean(axis=-1),
                   color=util.get_color_dict()[glom], marker='.', linestyle='-')

    for coh_ind, coh in enumerate(target_coherence):
        mean_val = normalized_response_amplitudes[g_ind, coh_ind, :].mean(axis=-1)
        err_val = normalized_response_amplitudes[g_ind, coh_ind, :].std(axis=-1) / np.sqrt(response_amplitudes.shape[-1])
        ax[g_ind].plot([coh, coh], [mean_val-err_val, mean_val+err_val],
                       color=util.get_color_dict()[glom], linestyle='-')

# Multiple comparisons test. Step down bonferroni
h, _, _, _ = multipletests(p_vals, alpha=0.05, method='holm')
for g_ind, glom in enumerate(included_gloms):
    if h[g_ind]:
        ax[g_ind].annotate('*', (0.5, 1.2), fontsize=18, ha='center')

ax[0].set_xlabel('Coherence')
ax[0].set_ylabel('Response (norm.)')
fh2.savefig(os.path.join(save_directory, 'coherence_tuning_curves.svg'), transparent=True)

# %% stim images
npoints = 300
w = 200
h = 200
steps = 10

# starting locations
x = np.random.uniform(low=steps, high=w-steps, size=npoints).astype(int)
y = np.random.uniform(low=steps, high=h-steps, size=npoints).astype(int)

mat_init = np.zeros((w, h))
mat_init[y, x] = 1
rand_step = mat_init.copy()
coh_step = mat_init.copy()

vel_x = np.ones(npoints).astype('int')
vel_y = 0
for step in range(1, steps):
    coh_step[y+step*vel_y, x+step*vel_x] = 1 - step/steps

direction = np.random.uniform(low=-np.pi, high=+np.pi, size=npoints)
vel_x = np.cos(direction)
vel_y = np.sin(direction)
for step in range(1, steps):
    rand_step[(y+step*vel_y).astype(int), (x+step*vel_x).astype(int)] = 1 - step/steps


fh3, ax3 = plt.subplots(2, 1, figsize=(3, 3), tight_layout=True)
cbar_ax = fh3.add_axes([.65, .1, .1, .8])
cbar_ax.set_axis_off()
[x.set_axis_off() for x in ax3.ravel()]
[x.set_xlim([50, 150]) for x in ax3.ravel()]
[x.set_ylim([50, 150]) for x in ax3.ravel()]
ax3[0].imshow(rand_step, cmap='Greys', vmin=-0.5, vmax=1.0)
ax3[0].set_title('Coherence = 0', fontsize=12)

ax3[1].imshow(coh_step, cmap='Greys', vmin=-0.5, vmax=1.0)
ax3[1].set_title('Coherence = 1', fontsize=12)

a = np.array([[0,1]])
img = plt.imshow(a, cmap="Greys_r")
plt.gca().set_visible(False)
cbar = fh3.colorbar(img, ax=cbar_ax, ticks=[0, 1])
cbar.ax.set_yticklabels(['t < 0', 't = 0'])

fh3.savefig(os.path.join(save_directory, 'coherence_stim_images.svg'), transparent=True)



 #%%
