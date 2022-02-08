import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import ants

from glom_pop import util, alignment
from visanalysis.analysis.shared_analysis import filterDataFiles
from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
transform_directory = os.path.join(base_dir, 'transforms', 'meanbrain_template')
experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'

target_gloms = ['LC4', 'LC9', 'LC18']

all_chat_responses = np.load(os.path.join(save_directory, 'chat_responses.npy'))
mean_chat_responses = np.load(os.path.join(save_directory, 'mean_chat_responses.npy'))
sem_chat_responses = np.load(os.path.join(save_directory, 'sem_chat_responses.npy'))
included_gloms = np.load(os.path.join(save_directory, 'included_gloms.npy'))
colors = np.load(os.path.join(save_directory, 'colors.npy'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))
glom_mask_2_meanbrain = ants.image_read(os.path.join(transform_directory, 'glom_mask_reg2meanbrain.nii')).numpy()
glom_mask_2_meanbrain = alignment.filterGlomMask_by_name(mask=glom_mask_2_meanbrain,
                                                         vpn_types=vpn_types,
                                                         included_gloms=included_gloms)

for g_ind, target_glom in enumerate(target_gloms):
    chat_glom_ind = np.where(included_gloms == target_glom)[0][0]
    glom_mask_val = vpn_types.loc[vpn_types.get('vpn_types') == target_glom, 'Unnamed: 0'].values[0]

    # Images & tuning response of split + chat
    fh0, ax0 = plt.subplots(2, 2, figsize=(6, 2),
                            gridspec_kw={'width_ratios': [1, 4], 'wspace': 0.01, 'hspace': 0.01})

    [plot_tools.cleanAxes(x) for x in ax0.ravel()]
    [x.set_ylim([-0.05, 0.35]) for x in ax0[:, 1]]

    ax0[0, 0].set_xlim([30, 230])
    ax0[0, 0].set_ylim([180, 0])
    ax0[1, 0].set_xlim([60, 160])
    ax0[1, 0].set_ylim([90, 0])

    # Response amps: split vs. chat scatter
    fh1, ax1 = plt.subplots(1, 1, figsize=(2, 1.75))

    util.makeGlomMap(ax=ax0[0, 0],
                     glom_map=glom_mask_2_meanbrain,
                     z_val=None,
                     highlight_vals=[glom_mask_val])

    # 0.5 micron pixels. 25 um scale bar
    plot_tools.addScaleBars(ax0[0, 0], dT=50, dF=0.0, T_value=35, F_value=175)

    target_series = filterDataFiles(experiment_file_directory,
                                    target_fly_metadata={'driver_1': target_glom},
                                    target_series_metadata={'protocol_ID': 'PanGlomSuite'},
                                    target_roi_series=['glom'])

    split_responses = []
    for s_ind, ser in enumerate(target_series):
        file_path = ser.get('file_name') + '.hdf5'
        series_number = ser.get('series')
        ID = volumetric_data.VolumetricDataObject(file_path,
                                                  series_number,
                                                  quiet=True)

        if s_ind == 0:
            glom_image = ID.getRoiResponses('glom').get('roi_image')
            ax0[1, 0].imshow(glom_image.mean(axis=-1).T, cmap='Greens')

        # Align responses
        time_vector, voxel_trial_matrix = ID.getTrialAlignedVoxelResponses(ID.getRoiResponses('glom').get('roi_response')[0], dff=True)
        mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(voxel_trial_matrix)
        n_stimuli = mean_voxel_response.shape[2]

        split_responses.append(mean_voxel_response)

        concat_resp = np.vstack(np.concatenate([mean_voxel_response[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))

    split_responses = np.vstack(split_responses)

    # shape = (n flies, concatenated time)
    individual_split_concat = np.concatenate([split_responses[:, :, x] for x in np.arange(len(unique_parameter_values)-2)], axis=1)
    individual_chat_concat = np.concatenate([all_chat_responses[chat_glom_ind, :, x, :] for x in np.arange(len(unique_parameter_values)-2)], axis=0).T
    concat_time = np.arange(0, individual_chat_concat.shape[-1]) * ID.getAcquisitionMetadata().get('sample_period')

    # (1) Plot chat vs split concat response for this glom type
    mean_chat = individual_chat_concat.mean(axis=0)
    err_chat = individual_chat_concat.std(axis=0) / np.sqrt(individual_chat_concat.shape[0])
    ax0[0, 1].plot(concat_time, mean_chat,
                   color=colors[chat_glom_ind, :], alpha=1.0, linewidth=1)
    ax0[0, 1].fill_between(concat_time,
                           mean_chat - err_chat,
                           mean_chat + err_chat,
                           color=colors[chat_glom_ind, :], alpha=0.5, linewidth=0)

    mean_split = individual_split_concat.mean(axis=0)
    err_split = individual_split_concat.std(axis=0) / np.sqrt(individual_split_concat.shape[0])
    ax0[1, 1].plot(concat_time, mean_split,
                   color='k', alpha=1.0, linewidth=1)
    ax0[1, 1].fill_between(concat_time,
                           mean_split - err_split,
                           mean_split + err_split,
                           color='k', alpha=0.5, linewidth=0)

    plot_tools.addScaleBars(ax0[0, 1], dT=5, dF=0.25, T_value=0, F_value=-0.045)

    # Compare mean response amplitudes
    split_amp = split_responses.max(axis=1)[:, :30]  # shape = (flies, stims)

    mean_split_amp = np.mean(split_amp, axis=0)
    sem_split_amp = (np.std(split_amp, axis=0) / np.sqrt(split_amp.shape[0]))

    chat_amp = all_chat_responses[chat_glom_ind, :, :, :].max(axis=0).T  # shape = (flies, stims)

    mean_chat_amp = np.mean(chat_amp, axis=0)
    sem_chat_amp = np.std(chat_amp, axis=0) / np.sqrt(chat_amp.shape[0])

    ax1.plot(mean_chat_amp, mean_split_amp, color='k', linestyle='none', marker='.')
    ax1.plot([mean_chat_amp-sem_chat_amp, mean_chat_amp+sem_chat_amp], [mean_split_amp, mean_split_amp], 'k-', alpha=0.5)
    ax1.plot([mean_chat_amp, mean_chat_amp], [mean_split_amp-sem_split_amp, mean_split_amp+sem_split_amp], 'k-', alpha=0.5)

    coef = np.polyfit(mean_chat_amp, mean_split_amp, 1)
    linfit = np.poly1d(coef)
    xx = [np.min([mean_chat_amp.min(), mean_split_amp.min()]), np.max([mean_chat_amp.max(), mean_split_amp.max()])]
    yy = linfit(xx)
    ax1.plot(xx, yy, 'k--')

    corr = np.corrcoef(mean_chat_amp, mean_split_amp)[1, 0]
    ax1.annotate('r = {:.2f}'.format(corr), (0, 1.1*mean_split_amp.max()))
    ax1.set_ylabel('dF/F, Split')
    ax1.set_xlabel('dF/F, ChAT')
    # TODO: annotation placement here

    mean_split_responses = split_responses.mean(axis=0)
    sem_split_responses = split_responses.std(axis=0) / split_responses.shape[0]

    fh0.savefig(os.path.join(save_directory, 'split_responses_{}.svg'.format(target_glom)), transparent=True)
    fh1.savefig(os.path.join(save_directory, 'split_amp_scatter_{}.svg'.format(target_glom)), transparent=True)


# %%



# %%
