import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

from glom_pop import dataio
from visanalysis.analysis import volumetric_data


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams.update({'font.sans-serif': 'Helvetica'})

experiment_file_directory = '/Users/mhturner/CurrentData'
save_directory = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/fig_panels'


# %% Across all flies: single trial stimulus identity encoding

path_to_yaml = '/Users/mhturner/Dropbox/ClandininLab/Analysis/glom_pop/glom_pop_data.yaml'

included_gloms = dataio.getIncludedGloms(path_to_yaml)
dataset = dataio.getDataset(path_to_yaml, dataset_id='pgs_tuning', only_included=True)

fh, ax = plt.subplots(3, 3, figsize=(12, 8))
ax = ax.ravel()

cmats = []
overall_performances = []
for s_ind, key in enumerate(dataset):
    experiment_file_name = key.split('_')[0]
    series_number = int(key.split('_')[1])

    file_path = os.path.join(experiment_file_directory, experiment_file_name + '.hdf5')
    ID = volumetric_data.VolumetricDataObject(file_path,
                                              series_number,
                                              quiet=True)

    # Load response data
    response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)
    vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

    # Only select gloms in included_gloms
    erm = []
    included_vals = []
    for g_ind, name in enumerate(included_gloms):
        pull_ind = np.where(name == names)[0][0]
        erm.append(response_data.get('epoch_response')[pull_ind, :, :])
        included_vals.append(vals[pull_ind])
    epoch_response_matrix = np.stack(erm, axis=0)
    included_vals = np.array(included_vals)

    parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
    unique_parameter_values = np.unique(np.array(parameter_values, dtype='object'))
    # Encode param sets to integers in order of unique_parameter_values
    # ref: https://stackoverflow.com/questions/38749305/labelencoder-order-of-fit-for-a-pandas-df
    df = pd.DataFrame(data=np.array(parameter_values, dtype='object'), columns=['params'])
    df['encoded'] = df['params'].apply(lambda x: list(unique_parameter_values).index(x))

    # Multinomial logistic regression model
    # shape = trials x time (concatenated glom responses)
    single_trial_responses = np.reshape(epoch_response_matrix, (-1, epoch_response_matrix.shape[2])).T

    # Filter trials to only include stims of interest
    # exclude last 2 (uniform flash)
    unique_parameter_values = unique_parameter_values[:30]
    keep_stims = np.arange(0, 30)
    keep_inds = np.where([x in keep_stims for x in df['encoded'].values])[0]

    X = single_trial_responses[keep_inds, :]
    y = df['encoded'].values[keep_inds]

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    y_test_all = []
    y_hat_all = []
    for it in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        y_test_all.append(y_test)
        y_hat_all.append(y_hat)

    y_test_all = np.hstack(y_test_all)
    y_hat_all = np.hstack(y_hat_all)
    performance = np.sum(y_hat_all == y_test_all) / y_test_all.shape[0]
    overall_performances.append(performance)

    cmat = confusion_matrix(y_test_all, y_hat_all, normalize='true')
    cmats.append(cmat)

    sns.heatmap(cmat, ax=ax[s_ind], vmin=0, vmax=1.0, cmap='Reds')
    ax[s_ind].set_xlabel('Predicted')
    ax[s_ind].set_ylabel('True')
    ax[s_ind].set_title('{:.2f}'.format(performance))
cmats = np.dstack(cmats)

# %%

col_names = [str(x) for x in unique_parameter_values]

mean_cmat = pd.DataFrame(data=cmats.mean(axis=-1), index=col_names, columns=col_names)

mean_cmat

# flies x params
diag_values = np.vstack([np.diag(cmats[:, :, x]) for x in range(cmats.shape[-1])])
diag_values.shape
mean_diag = np.mean(diag_values, axis=0)
err_diag = np.std(diag_values, axis=0) / np.sqrt(diag_values.shape[0])

# Performance per stim type
fh0, ax0 = plt.subplots(1, 1, figsize=(6, 3))
ax0.plot(col_names, mean_diag, 'ko')
for d_ind in range(len(col_names)):
    ax0.plot([d_ind, d_ind], [mean_diag[d_ind]-err_diag[d_ind], mean_diag[d_ind]+err_diag[d_ind]], 'k-')

ax0.plot([0, len(model.classes_)], [1/len(model.classes_), 1/len(model.classes_)], 'k--')
ax0.set_xticks([])
ax0.set_ylim([-0.1, 1])
ax0.set_ylabel('Performance')
ax0.tick_params(axis='y', labelsize=11)
ax0.tick_params(axis='x', labelsize=11, rotation=90)

# Confusion matrix
fh1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(mean_cmat, ax=ax1, vmin=0, vmax=1.0, cmap='Reds')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_xticks([])
ax1.set_yticks([])

fh0.savefig(os.path.join(save_directory, 'single_trial_performance.pdf'))
fh1.savefig(os.path.join(save_directory, 'single_trial_confusion.pdf'))




# %%
# Weights: classes x features
weights = model.coef_

sns.heatmap(weights, vmin=-1, vmax=+1, cmap='RdBu_r')



# %%
