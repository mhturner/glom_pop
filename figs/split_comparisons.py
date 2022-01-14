import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import ants

from glom_pop import util, alignment
from visanalysis.analysis.shared_analysis import filterDataFiles
from visanalysis.analysis import volumetric_data
from visanalysis.util import plot_tools
from scipy.stats import zscore
import seaborn as sns


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

base_dir = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop'
transform_directory = os.path.join(base_dir, 'transforms', 'meanbrain_template')
experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'

target_gloms = ['LC4', 'LC9', 'LC18']
yoffset = 0.0  # Split vs. Chat. Share a y axis

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

# PGS mean responses, split & chat on same axis
fh0, ax0 = plt.subplots(len(target_gloms), 30, figsize=(6, 4))
[plot_tools.cleanAxes(x) for x in ax0.ravel()]
[x.set_ylim([-0.05, 0.6]) for x in ax0.ravel()]
fh0.subplots_adjust(wspace=0.00, hspace=0.00)

# Images of gloms in glom map and example split image
fh1, ax1 = plt.subplots(len(target_gloms)*2, 1, figsize=(1, 4))
[x.set_xlim([30, 230]) for x in ax1[1::2]]
[x.set_ylim([180, 0]) for x in ax1[1::2]]
[x.set_xlim([60, 160]) for x in ax1[::2]]
[x.set_ylim([90, 0]) for x in ax1[::2]]
[x.set_axis_off() for x in ax1.ravel()]

# Response amps: split vs. chat scatter
fh2, ax2 = plt.subplots(1, len(target_gloms), figsize=(6, 1.75))

all_split_mean = []
all_chat_mean = []

for g_ind, target_glom in enumerate(target_gloms):
    chat_glom_ind = np.where(included_gloms == target_glom)[0][0]
    glom_mask_val = vpn_types.loc[vpn_types.get('vpn_types') == target_glom, 'Unnamed: 0'].values[0]

    util.makeGlomMap(ax=ax1[g_ind*2 + 1],
                     glom_map=glom_mask_2_meanbrain,
                     z_val=None,
                     highlight_vals=[glom_mask_val])
    if g_ind == 0:
        plot_tools.addScaleBars(ax1[g_ind*2], dT=25, dF=0.0, T_value=65, F_value=90)

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
            ax1[g_ind*2].imshow(glom_image.mean(axis=-1).T, cmap='Greens')

        # Align responses
        time_vector, voxel_trial_matrix = ID.getTrialAlignedVoxelResponses(ID.getRoiResponses('glom').get('roi_response')[0], dff=True)
        mean_voxel_response, unique_parameter_values, _, response_amp, trial_response_amp, _ = ID.getMeanBrainByStimulus(voxel_trial_matrix)
        n_stimuli = mean_voxel_response.shape[2]

        split_responses.append(mean_voxel_response)

        concat_resp = np.vstack(np.concatenate([mean_voxel_response[:, :, x] for x in np.arange(len(unique_parameter_values))], axis=1))

    split_responses = np.vstack(split_responses)

    # Append concatenated mean responses (mean across flies)
    split_concat = np.vstack(np.concatenate([split_responses.mean(axis=0)[:, x] for x in np.arange(len(unique_parameter_values)-2)], axis=0))
    all_split_mean.append(split_concat)

    chat_concat = np.vstack(np.concatenate([mean_chat_responses[chat_glom_ind, :, x] for x in np.arange(len(unique_parameter_values)-2)], axis=0))
    all_chat_mean.append(chat_concat)
    mean_concat = np.vstack(np.concatenate([split_responses.mean(axis=0)[:, x] for x in np.arange(len(unique_parameter_values))], axis=0))

    # shape = (n flies, concatenated time)
    individual_split_concat = np.concatenate([split_responses[:, :, x] for x in np.arange(len(unique_parameter_values)-2)], axis=1)
    individual_chat_concat = np.concatenate([all_chat_responses[chat_glom_ind, :, x, :] for x in np.arange(len(unique_parameter_values)-2)], axis=0).T

    # Compare mean response amplitudes
    split_amp = split_responses.max(axis=1)[:, :30]  # shape = (flies, stims)
    # split_amp = zscore(split_amp, axis=1)

    mean_split_amp = np.mean(split_amp, axis=0)
    sem_split_amp = (np.std(split_amp, axis=0) / np.sqrt(split_amp.shape[0]))

    chat_amp = all_chat_responses[chat_glom_ind, :, :, :].max(axis=0).T  # shape = (flies, stims)
    # chat_amp = zscore(chat_amp, axis=1)

    mean_chat_amp = np.mean(chat_amp, axis=0)
    sem_chat_amp = np.std(chat_amp, axis=0) / np.sqrt(chat_amp.shape[0])

    ax2[g_ind].plot(mean_chat_amp, mean_split_amp, color='k', linestyle='none', marker='.')
    ax2[g_ind].plot([mean_chat_amp-sem_chat_amp, mean_chat_amp+sem_chat_amp], [mean_split_amp, mean_split_amp], 'k-')
    ax2[g_ind].plot([mean_chat_amp, mean_chat_amp], [mean_split_amp-sem_split_amp, mean_split_amp+sem_split_amp], 'k-')

    coef = np.polyfit(mean_chat_amp, mean_split_amp, 1)
    linfit = np.poly1d(coef)
    xx = [np.min([mean_chat_amp.min(), mean_split_amp.min()]), np.max([mean_chat_amp.max(), mean_split_amp.max()])]
    ax2[g_ind].plot(xx, linfit(xx), 'k--')

    corr = np.corrcoef(mean_chat_amp, mean_split_amp)[1, 0]
    ax2[g_ind].set_title(target_glom)
    ax2[g_ind].annotate('r = {:.2f}'.format(corr), (0, 1.1*mean_split_amp.max()))

    if g_ind == 0:
        ax2[g_ind].set_ylabel('dF/F, Split')
    elif g_ind == 1:
        ax2[g_ind].set_xlabel('dF/F, ChAT')

    # intra-individual corr for this split
    intra_ind_corr = pd.DataFrame(np.max(split_responses, axis=1).T).corr().to_numpy()[np.triu_indices(split_responses.shape[0], k=1)]

    mean_split_responses = split_responses.mean(axis=0)
    sem_split_responses = split_responses.std(axis=0) / split_responses.shape[0]

    for u_ind, un in enumerate(unique_parameter_values[:-2]):
        #  Plot mean +/- sem split responses
        ax0[g_ind, u_ind].plot(time_vector, yoffset + mean_split_responses[:, u_ind],
                               color='k', alpha=1.0, linewidth=2)
        ax0[g_ind, u_ind].fill_between(time_vector,
                                       yoffset + mean_split_responses[:, u_ind] - sem_split_responses[:, u_ind],
                                       yoffset + mean_split_responses[:, u_ind] + sem_split_responses[:, u_ind],
                                       color='k', alpha=0.5, linewidth=0)

        #  Plot mean +/- sem chat responses
        ax0[g_ind, u_ind].plot(time_vector, mean_chat_responses[chat_glom_ind, :, u_ind],
                               color=colors[chat_glom_ind, :], alpha=1.0, linewidth=2)
        ax0[g_ind, u_ind].fill_between(time_vector,
                                       mean_chat_responses[chat_glom_ind, :, u_ind] - sem_chat_responses[chat_glom_ind, :, u_ind],
                                       mean_chat_responses[chat_glom_ind, :, u_ind] + sem_chat_responses[chat_glom_ind, :, u_ind],
                                       color=colors[chat_glom_ind, :], alpha=1.0, linewidth=0)

        if u_ind == 0:
            ax0[g_ind, u_ind].annotate(target_glom, (1, 0.5), fontsize=14, fontweight='bold')


plot_tools.addScaleBars(ax0[0, 0], dT=5, dF=0.25, T_value=0, F_value=yoffset-0.08)

fh0.savefig(os.path.join(save_directory, 'split_responses.svg'), transparent=True)
fh1.savefig(os.path.join(save_directory, 'split_glom_images.svg'), transparent=True)
fh2.savefig(os.path.join(save_directory, 'split_amp_scatter.svg'), transparent=True)


# %%

# %%



# %%
