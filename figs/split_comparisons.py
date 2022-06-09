import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import ants
from glom_pop import util, alignment, dataio
from visanalysis.analysis.shared_analysis import filterDataFiles
from visanalysis.analysis import imaging_data
from visanalysis.util import plot_tools

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']
transform_directory = os.path.join(sync_dir, 'transforms', 'meanbrain_template')

target_gloms = ['LC18', 'LC9', 'LC4']

chat_response_amplitudes = np.load(os.path.join(save_directory, 'chat_response_amplitudes.npy'))
all_chat_responses = np.load(os.path.join(save_directory, 'chat_responses.npy'))
mean_chat_responses = np.load(os.path.join(save_directory, 'mean_chat_responses.npy'))
sem_chat_responses = np.load(os.path.join(save_directory, 'sem_chat_responses.npy'))
included_gloms = dataio.get_included_gloms()

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(sync_dir, 'template_brain', 'vpn_types.csv'))
glom_mask_2_meanbrain = ants.image_read(os.path.join(transform_directory, 'glom_mask_reg2meanbrain.nii')).numpy()
glom_mask_2_meanbrain = alignment.filter_glom_mask_by_name(mask=glom_mask_2_meanbrain,
                                                           vpn_types=vpn_types,
                                                           included_gloms=included_gloms)
fh3, ax3 = plt.subplots(3, 1, figsize=(7, 4))
[x.set_axis_off() for x in ax3.ravel()]
[x.set_ylim([-0.1, 0.45]) for x in ax3.ravel()]

# Response amps: split vs. chat scatter
fh1, ax1 = plt.subplots(1, 3, figsize=(4.5, 1.75), tight_layout=True)
for g_ind, target_glom in enumerate(target_gloms):
    chat_glom_ind = np.where(np.array(included_gloms) == target_glom)[0][0]
    # Images of split + chat
    fh0, ax0 = plt.subplots(1, 2, figsize=(1.5, 0.75),
                            gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

    [plot_tools.cleanAxes(x) for x in ax0.ravel()]
    ax0[0].set_xlim([30, 230])
    ax0[0].set_ylim([180, 0])
    ax0[1].set_xlim([60, 160])
    ax0[1].set_ylim([90, 0])

    util.make_glom_map(ax=ax0[0],
                       glom_map=glom_mask_2_meanbrain,
                       z_val=None,
                       highlight_names=[target_glom])

    # 0.5 micron pixels. 25 um scale bar
    plot_tools.addScaleBars(ax0[0], dT=50, dF=0.0, T_value=35, F_value=175)

    target_series = filterDataFiles(os.path.join(sync_dir, 'datafiles'),
                                    target_fly_metadata={'driver_1': target_glom},
                                    target_series_metadata={'protocol_ID': 'PanGlomSuite',
                                                            'include_in_analysis': True},
                                    target_roi_series=['glom'])

    split_responses = []
    split_response_amplitudes = []

    for s_ind, ser in enumerate(target_series):
        file_path = ser.get('file_name') + '.hdf5'
        file_name = os.path.split(ser['file_name'])[-1]
        series_number = ser.get('series')
        # print('------')
        # print(file_name)
        # print(series_number)
        # print('------')
        ID = imaging_data.ImagingDataObject(file_path,
                                            series_number,
                                            quiet=True)

        if s_ind == 0:
            glom_image = ID.getRoiResponses('glom').get('roi_image')
            ax0[1].imshow(glom_image.mean(axis=-1).T, cmap='Greys_r')

        # Align responses
        roi_data = ID.getRoiResponses('glom')
        unique_parameter_values, mean_response, _, _ = ID.getTrialAverages(roi_data['epoch_response'])

        response_amp = ID.getResponseAmplitude(mean_response, metric='max')

        split_response_amplitudes.append(response_amp)
        split_responses.append(mean_response)

    split_responses = np.vstack(split_responses)  # flies x stim x time
    split_response_amplitudes = np.vstack(split_response_amplitudes)[:, :30]  # flies x stim

    chat_response_amplitudes = np.vstack([ID.getResponseAmplitude(all_chat_responses[chat_glom_ind, :, :, x],
                                                                  metric='max') for x in range(all_chat_responses.shape[-1])])

    # shape = (n flies, concatenated time)
    individual_split_concat = np.concatenate([split_responses[:, x, :] for x in np.arange(len(unique_parameter_values)-2)], axis=1)
    individual_chat_concat = np.concatenate([all_chat_responses[chat_glom_ind, x, :, :] for x in np.arange(len(unique_parameter_values)-2)], axis=0).T
    concat_time = np.arange(0, individual_chat_concat.shape[-1]) * ID.getAcquisitionMetadata().get('sample_period')

    # (1) Plot chat vs split concat response for this glom type
    # mean_chat = individual_chat_concat.mean(axis=0)
    # err_chat = individual_chat_concat.std(axis=0) / np.sqrt(individual_chat_concat.shape[0])
    # ax0[0, 1].plot(concat_time, mean_chat,
    #                color=util.get_color_dict().get(target_glom), alpha=1.0, linewidth=1)
    # ax0[0, 1].fill_between(concat_time,
    #                        mean_chat - err_chat,
    #                        mean_chat + err_chat,
    #                        color=util.get_color_dict().get(target_glom), alpha=0.5, linewidth=0)
    if False:  # QC
        fh, ax = plt.subplots(10, 2, figsize=(8, 8))
        for i in range(individual_split_concat.shape[0]):
            ax[i, 0].plot(individual_split_concat[i, :], 'k')
        for i in range(individual_chat_concat.shape[0]):
            ax[i, 1].plot(individual_chat_concat[i, :], 'b')

    # mean_split = individual_split_concat.mean(axis=0)
    # err_split = individual_split_concat.std(axis=0) / np.sqrt(individual_split_concat.shape[0])
    # ax0[1, 1].plot(concat_time, mean_split,
    #                color='k', alpha=1.0, linewidth=1)
    # ax0[1, 1].fill_between(concat_time,
    #                        mean_split - err_split,
    #                        mean_split + err_split,
    #                        color='k', alpha=0.5, linewidth=0)

    # Compare mean response amplitudes
    split_amp = split_response_amplitudes  # shape = (flies, stims)
    mean_split_amp = np.mean(split_amp, axis=0)
    sem_split_amp = (np.std(split_amp, axis=0) / np.sqrt(split_amp.shape[0]))

    chat_amp = chat_response_amplitudes  # shape = (flies, stims)
    mean_chat_amp = np.mean(chat_amp, axis=0)
    sem_chat_amp = np.std(chat_amp, axis=0) / np.sqrt(chat_amp.shape[0])

    ax1[g_ind].plot(mean_chat_amp, mean_split_amp, color='k', linestyle='none', marker='.')
    ax1[g_ind].plot([mean_chat_amp-sem_chat_amp, mean_chat_amp+sem_chat_amp], [mean_split_amp, mean_split_amp], 'k-', alpha=0.5)
    ax1[g_ind].plot([mean_chat_amp, mean_chat_amp], [mean_split_amp-sem_split_amp, mean_split_amp+sem_split_amp], 'k-', alpha=0.5)

    coef = np.polyfit(mean_chat_amp, mean_split_amp, 1)
    linfit = np.poly1d(coef)
    xx = [np.min([mean_chat_amp.min(), mean_split_amp.min()]), np.max([mean_chat_amp.max(), mean_split_amp.max()])]
    yy = linfit(xx)
    ax1[g_ind].plot(xx, yy, 'k--')

    corr = np.corrcoef(mean_chat_amp, mean_split_amp)[1, 0]
    ax1[g_ind].annotate('r = {:.2f}'.format(corr), (0.55*np.max(xx), 0.1*np.max(yy)))
    ax1[g_ind].set_xlim(left=0)
    ax1[g_ind].set_ylim(bottom=0)
    ax1[g_ind].spines['top'].set_visible(False)
    ax1[g_ind].spines['right'].set_visible(False)

    if g_ind == 1:
        ax1[g_ind].set_xlabel('ChAT response (dF/F)')
    elif g_ind == 0:
        ax1[g_ind].set_ylabel('Split-Gal4 \nresponse (dF/F)')

    mean_split_responses = split_responses.mean(axis=0)
    sem_split_responses = split_responses.std(axis=0) / split_responses.shape[0]

    ax3[g_ind].plot(concat_time,
                    np.mean(individual_split_concat, axis=0),
                    color='k',
                    label='Split-Gal4' if g_ind == 0 else None)
    ax3[g_ind].plot(concat_time,
                    np.mean(individual_chat_concat, axis=0),
                    color=util.get_color_dict()[target_glom],
                    label='ChAT-Gal4' if g_ind == 0 else None)
    if g_ind == 2:
        plot_tools.addScaleBars(ax3[g_ind], dT=2, dF=0.25, T_value=0, F_value=-0.05)

    # for s in range(30):
    #     ax3[g_ind, s].plot(roi_data['time_vector'], np.mean(split_responses, axis=0)[s, :], color='k')
    #     ax3[g_ind, s].plot(roi_data['time_vector'], np.mean(all_chat_responses[chat_glom_ind, ...], axis=-1)[s, :], color=util.get_color_dict()[target_glom])
    #     if s == 0:
    #         if g_ind == 2:
    #             plot_tools.addScaleBars(ax3[g_ind, s], dT=2, dF=0.25, T_value=0, F_value=-0.05)

    fh0.savefig(os.path.join(save_directory, 'split_images_{}.svg'.format(target_glom)), transparent=True)

fh3.legend()
fh1.savefig(os.path.join(save_directory, 'split_amp_scatter.svg'), transparent=True)
fh3.savefig(os.path.join(save_directory, 'split_amp_traces.svg'), transparent=True)
# %%


# %%
