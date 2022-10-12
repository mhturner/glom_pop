from visanalysis.analysis import imaging_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
from ast import literal_eval as make_tuple
from scipy.stats import ttest_1samp

from glom_pop import dataio, util

util.config_matplotlib()

sync_dir = dataio.get_config_file()['sync_dir']
save_directory = dataio.get_config_file()['save_directory']

leaves = np.load(os.path.join(save_directory, 'cluster_leaves_list.npy'))
included_gloms = dataio.get_included_gloms()
# sort by dendrogram leaves ordering
included_gloms = np.array(included_gloms)[leaves]
# Include only small spot responder gloms
included_gloms = ['LC11', 'LC21', 'LC18', 'LC6', 'LC26', 'LC17', 'LC12', 'LC15']
included_vals = dataio.get_glom_vals_from_names(included_gloms)

eg_series = ('2021-08-11', 2)

matching_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                  target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                       'indicator_1': 'Syt1GCaMP6f',
                                                                       'indicator_2': 'TdTomato'},
                                                  target_series_metadata={'protocol_ID': 'ForestRandomWalk',
                                                                          'include_in_analysis': True},
                                                  target_groups=['aligned_response'])

# %%

peak_ls = []
peak_rs = []
peak_vrs = []

for s_ind, series in enumerate(matching_series):
    series_number = series['series']
    file_path = series['file_name'] + '.hdf5'
    file_name = os.path.split(series['file_name'])[-1]
    fly_id = series['fly_id']
    anatomical_brain = series['anatomical_brain']
    ID_vr = imaging_data.ImagingDataObject(file_path,
                                           series_number,
                                           quiet=True)

    # get VR response data:
    response_data_vr = dataio.load_responses(ID_vr, response_set_name='glom')
    epoch_response_matrix_vr = dataio.filter_epoch_response_matrix(response_data_vr, included_vals)
    # Align responses
    unique_parameter_values_vr, mean_response_vr, _, _ = ID_vr.getTrialAverages(epoch_response_matrix_vr, parameter_key='current_trajectory_index')
    peak_vr = ID_vr.getResponseAmplitude(mean_response_vr)

    # associated PGS series from that fly: L & R dark bar for comparison...
    pgs_series = shared_analysis.filterDataFiles(data_directory=os.path.join(sync_dir, 'datafiles'),
                                                 target_fly_metadata={'driver_1': 'ChAT-T2A',
                                                                      'indicator_1': 'Syt1GCaMP6f',
                                                                      'indicator_2': 'TdTomato',
                                                                      'fly_id': fly_id},
                                                 target_series_metadata={'protocol_ID': 'PanGlomSuite',
                                                                         'include_in_analysis': True,
                                                                         'anatomical_brain': anatomical_brain},
                                                 target_groups=['aligned_response'],
                                                 quiet=True)
    assert len(pgs_series) == 1, '{}: !=1 matching pgs series'.format(s_ind)

    pgs_series = pgs_series[0]
    ID_pgs = imaging_data.ImagingDataObject(pgs_series['file_name']+'.hdf5',
                                            pgs_series['series'],
                                            quiet=True)

    response_data = dataio.load_responses(ID_pgs, response_set_name='glom', get_voxel_responses=False)
    epoch_response_matrix = dataio.filter_epoch_response_matrix(response_data, included_vals)

    query_l = {'component_stim_type': 'MovingRectangle', 'current_intensity': 0, 'current_angle': 0}
    trials_l = shared_analysis.filterTrials(epoch_response_matrix, ID_pgs, query_l, return_inds=False)
    mean_pgs_l = np.mean(trials_l, axis=1)
    peak_l = ID_pgs.getResponseAmplitude(mean_pgs_l)

    query_r = {'component_stim_type': 'MovingRectangle', 'current_intensity': 0, 'current_angle': 180}
    trials_r = shared_analysis.filterTrials(epoch_response_matrix, ID_pgs, query_r, return_inds=False)
    mean_pgs_r = np.mean(trials_r, axis=1)
    peak_r = ID_pgs.getResponseAmplitude(mean_pgs_r)

    peak_ls.append(peak_l)
    peak_rs.append(peak_r)
    peak_vrs.append(peak_vr)

    if np.logical_and(file_name == eg_series[0], series_number == eg_series[1]):
        traj_ind = 3

        fh0, ax0 = plt.subplots(2+len(included_gloms), 3, figsize=(2, 5),
                                gridspec_kw={'width_ratios': [4, 1, 1],
                                             'wspace': 0.01, 'hspace': 0.01})
        [util.clean_axes(x) for x in ax0[2:, :].ravel()]
        [util.clean_axes(x) for x in ax0[0:2, -2:].ravel()]
        [x.set_ylim([-0.15, 0.9]) for x in ax0[1:, :].ravel()]

        # Plot VR trajectory traces
        epoch_parameters = ID_vr.getEpochParameters()
        query = {'current_trajectory_index': traj_ind}
        trials, trial_inds = shared_analysis.filterTrials(response_data.get('epoch_response'), ID_vr, query, return_inds=True)

        x_tv = make_tuple(epoch_parameters[trial_inds[0]].get('fly_x_trajectory', epoch_parameters[trial_inds[0]].get('stim0_fly_x_trajectory'))).get('tv_pairs')
        y_tv = make_tuple(epoch_parameters[trial_inds[0]].get('fly_y_trajectory', epoch_parameters[trial_inds[0]].get('stim0_fly_y_trajectory'))).get('tv_pairs')
        theta_tv = make_tuple(epoch_parameters[trial_inds[0]].get('fly_theta_trajectory', epoch_parameters[trial_inds[0]].get('stim0_fly_theta_trajectory'))).get('tv_pairs')

        trajectory_time = [tv[0] for tv in x_tv] + ID_vr.getRunParameters().get('pre_time')
        sample_period = np.diff(trajectory_time)[0]  # sec
        x_position = np.array([tv[1] for tv in x_tv])
        y_position = np.array([tv[1] for tv in y_tv])
        theta = np.array([tv[1] for tv in theta_tv])

        # calc velocity. Translational and rotational
        v_x = np.diff(x_position)
        v_y = np.diff(y_position)
        v_trans = np.linalg.norm(np.vstack([v_x, v_y]), axis=0)  # m / sample period
        v_trans = 100 * v_trans / sample_period  # -> cm / sec

        ax0[0, 0].axhline(y=0, color='k', alpha=0.5)
        ax0[0, 0].plot(trajectory_time[:-1], v_trans, 'k', label='Translational')
        ax0[0, 0].set_ylim([-0.5, 3.5])
        # ax0[0, 0].set_ylabel('Trans.\n(cm/s)')
        ax0[0, 0].spines['top'].set_visible(False)
        ax0[0, 0].spines['right'].set_visible(False)
        ax0[0, 0].spines['bottom'].set_visible(False)

        v_ang = np.diff(theta) / sample_period  # deg / sample period -> deg/sec
        ax0[1, 0].axhline(y=0, color='k', alpha=0.5)
        ax0[1, 0].plot(trajectory_time[:-1], v_ang, color='b', label='theta')
        ax0[1, 0].set_ylim([-200, 200])
        ax0[1, 0].set_yticks([-100, 100])
        # ax0[1, 0].set_ylabel('Ang.\n($^\circ$/s)')
        ax0[1, 0].spines['top'].set_visible(False)
        ax0[1, 0].spines['right'].set_visible(False)
        ax0[1, 0].spines['bottom'].set_visible(False)

        # else:
        #     util.clean_axes(ax0[0, un])
        #     util.clean_axes(ax0[1, un])
        # plot glom responses to VR and bars
        for g_ind, glom in enumerate(included_gloms):
            # Plot VR responses
            ax0[g_ind+2, 0].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax0[g_ind+2, 0].plot(response_data_vr['time_vector'],
                                 mean_response_vr[g_ind, traj_ind, :], color=util.get_color_dict()[glom])
            if g_ind == 0:
                plot_tools.addScaleBars(ax0[g_ind+2, 0], dT=5, dF=0.25, T_value=-0.5, F_value=-0.12)

            # Plot L & R bar responses
            ax0[g_ind+2, 1].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax0[g_ind+2, 1].plot(response_data['time_vector'],
                                 mean_pgs_l[g_ind, :], color=util.get_color_dict()[glom])

            ax0[g_ind+2, 2].axhline(0, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax0[g_ind+2, 2].plot(response_data['time_vector'],
                                 mean_pgs_r[g_ind, :], color=util.get_color_dict()[glom])

            if g_ind == 0:
                plot_tools.addScaleBars(ax0[g_ind+2, 1], dT=5, dF=0.25, T_value=-0.5, F_value=-0.12)

        # top-down view of trajectory
        x_position = 100*np.array([tv[1] for tv in x_tv])  # -> cm
        y_position = 100*np.array([tv[1] for tv in y_tv])  # -> cm
        theta = np.array([tv[1] for tv in theta_tv]) + 90  # adjust for flystim screen coords
        quiver_stride = 250
        fh2, ax2 = plt.subplots(1, 1, figsize=(2, 2))
        ax2.plot(x_position, y_position, 'k', linewidth=1, alpha=0.5)
        ax2.plot(x_position[0], y_position[0], 'go')
        ax2.plot(x_position[-1], y_position[-1], 'ro')
        ax2.quiver(x_position[::quiver_stride],
                   y_position[::quiver_stride],
                   np.cos(np.radians(theta[::quiver_stride])),
                   np.sin(np.radians(theta[::quiver_stride])),
                   color='b')
        dx = 5  # cm
        ax2.plot([-5, -5+dx], [-2.5, -2.5],
                 color='k',
                 linewidth=2)
        ax2.axis('equal')
        ax2.set_axis_off()
        fh2.savefig(os.path.join(save_directory, 'vr_eg_traj.svg'))

peak_ls = np.vstack(peak_ls)  # flies x gloms
peak_rs = np.vstack(peak_rs)
peak_vrs = np.dstack(peak_vrs)  # gloms x trajectories x flies

# Avg L & R
peak_bar = (peak_ls + peak_rs) / 2


fh0.savefig(os.path.join(save_directory, 'vr_eg_traces.svg'))


# %% Histogram of peak response for each trajectory

fh1, ax1 = plt.subplots(peak_vrs.shape[0], 1, figsize=(1.5, 3.9))

for g_ind, glom in enumerate(included_gloms):
    ct, bn = np.histogram(peak_vrs[g_ind, :, :].ravel(),
                          bins=np.linspace(0, np.nanmax(peak_vrs), 40),
                          density=False)
    bn_ctr = bn[:-1] + np.diff(bn)[0]
    bn_prob = ct / np.sum(ct)

    ax1[g_ind].fill_between(bn_ctr, bn_prob, color=util.get_color_dict()[glom])
    mean_bar = np.nanmean(peak_bar[:, g_ind])
    ax1[g_ind].axvline(x=mean_bar, color='k', linewidth=2)
    ax1[g_ind].set_xlim([0, 0.6])
    ax1[g_ind].set_ylim([0, 0.25])
    # ttest: mean across trajectories, vs. bar response from the same glom/fly
    h, p = ttest_1samp(peak_vrs[g_ind, ...].ravel(), popmean=mean_bar)
    # if p < (0.05 / len(included_gloms)):
    #     ax[g_ind].annotate('*', (mean_bar+0.01, 0.15), fontsize=18)
[x.set_xticks([]) for x in ax1[:-1]]
[x.set_yticks([]) for x in ax1[:-1]]
ax1[-1].set_yticks([0, 0.2])


[x.spines['top'].set_visible(False) for x in ax1]
[x.spines['right'].set_visible(False) for x in ax1]

fh1.supxlabel('Peak response \n amplitude (dF/F)')
fh1.supylabel('Fraction of VR trajectories')


fh1.savefig(os.path.join(save_directory, 'vr_summary_hists.svg'))


# %%
