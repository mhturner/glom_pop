from visanalysis.analysis import volumetric_data, shared_analysis
from visanalysis.util import plot_tools
import matplotlib.pyplot as plt
import numpy as np
import os
import colorcet as cc
from ast import literal_eval as make_tuple
import seaborn as sns
import pandas as pd

from glom_pop import dataio, util


experiment_file_directory = '/Users/mhturner/CurrentData'
# save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'

path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'
included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='tower_distance', only_included=True)

parameter_keys = ['current_forward_velocity', 'current_tower_diameter', 'current_tower_xoffset']

all_resp = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')

    # ImagingDataObject wants a path to an hdf5 file and a series number from that file
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    epoch_parameters = ID.getEpochParameters()

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom')

    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

    meanbrain_red = response_data.get('meanbrain')[..., 0]
    meanbrain_green = response_data.get('meanbrain')[..., 1]

    # Only select gloms in included_gloms
    erm = []
    included_vals = []
    for g_ind, name in enumerate(included_gloms):
        pull_ind = np.where(name==names)[0][0]
        erm.append(response_data.get('epoch_response')[pull_ind, :, :])
        included_vals.append(vals[pull_ind])
    # epoch_response_matrix = n gloms, time, n trials
    epoch_response_matrix = np.stack(erm, axis=0)
    included_vals = np.array(included_vals)
    epoch_response_matrix.shape

    parameter_values = [list(param_dict.values()) for param_dict in ID.getEpochParameterDicts()]
    unique_parameter_values = np.unique(np.array(parameter_values), axis=0)

    tower_diameter = ID.getRunParameters().get('tower_diameter')
    tower_xoffset = ID.getRunParameters().get('tower_xoffset')

    tower_diameter
    tower_xoffset


    fh, ax = plt.subplots(len(tower_diameter), len(tower_xoffset), figsize=(8, 8))
    [x.set_axis_off() for x in ax.ravel()]
    [x.set_ylim([-0.25, 1.0]) for x in ax.ravel()]
    for u_ind, param_set in enumerate(unique_parameter_values):
        # Yank out trials for this param combo
        query = {'current_forward_velocity': param_set[0],
                 'current_tower_diameter': param_set[1],
                 'current_tower_xoffset': param_set[2]}
        # Reshape trials to n gloms, n trials, time
        trials = shared_analysis.filterTrials(np.swapaxes(epoch_response_matrix, 1, 2), ID, query)
        trials.shape
        mean_resp = trials.mean(axis=1)

        diameter_ind = np.where(param_set[1] == tower_diameter)[0][0]
        offset_ind = np.where(param_set[2] == tower_xoffset)[0][0]

        theta = np.abs(np.rad2deg(2*np.arctan((param_set[1]/2) / param_set[2])))
        for g_ind, glom in enumerate(included_gloms):
            ax[diameter_ind, offset_ind].plot(mean_resp[g_ind, :],
                                              linestyle='-', label=glom if u_ind==0 else '_'+glom, color=ID.colors[g_ind])

        ax[diameter_ind, offset_ind].annotate('{:.0f}'.format(theta), (1, 0.5))

        if u_ind==0:
            plot_tools.addScaleBars(ax[0, 0], dT=5, dF=0.25, F_value=-0.2, T_value=-0.5)

        if offset_ind==0:
            ax[diameter_ind, offset_ind].set_axis_on()
            ax[diameter_ind, offset_ind].set_ylabel(int(param_set[1] * 1e2))
            ax[diameter_ind, offset_ind].set_xticks([])
            ax[diameter_ind, offset_ind].set_yticks([])
            ax[diameter_ind, offset_ind].spines['right'].set_visible(False)
            ax[diameter_ind, offset_ind].spines['top'].set_visible(False)
            ax[diameter_ind, offset_ind].spines['left'].set_visible(False)
            ax[diameter_ind, offset_ind].spines['bottom'].set_visible(False)

        if diameter_ind==0:
            ax[diameter_ind, offset_ind].set_title(int(param_set[2] * 1e2))

# %%
