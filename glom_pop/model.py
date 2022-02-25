"""

"""
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import colorcet as cc
from scipy.signal import detrend
from scipy.stats import zscore

from visanalysis.analysis import imaging_data
from glom_pop import dataio


class SingleTrialEncoding():
    def __init__(self, dataset, included_vals, experiment_file_directory='/Users/mhturner/CurrentData'):
        self.experiment_file_directory = experiment_file_directory
        self.dataset = dataset
        self.included_vals = included_vals

    def evaluate_performance(self, model_type='LogReg', iterations=20, pull_eg=0, classify_on_amplitude=False, random_state=None):

        self.cmats = []
        self.overall_performances = []
        for s_ind, key in enumerate(self.dataset):
            experiment_file_name = key.split('_')[0]
            series_number = int(key.split('_')[1])

            file_path = os.path.join(self.experiment_file_directory, experiment_file_name + '.hdf5')
            ID = imaging_data.ImagingDataObject(file_path,
                                                series_number,
                                                quiet=True)

            # Load response data
            response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

            # Only select gloms in included_vals
            glom_size_threshold = 10
            # response_matrix: shape=(gloms, time)
            response_matrix = np.zeros((len(self.included_vals), response_data.get('response').shape[1]))
            for val_ind, included_val in enumerate(self.included_vals):
                new_glom_size = np.sum(response_data.get('mask') == included_val)

                if new_glom_size > glom_size_threshold:
                    pull_ind = np.where(included_val == response_data.get('mask_vals'))[0][0]
                    response_matrix[val_ind, :] = response_data.get('response')[pull_ind, :]
                else:  # Exclude because this glom, in this fly, is too tiny
                    pass
            response_matrix = np.stack(response_matrix, axis=0)

            # Detrend (remove ~linear bleach) & z-score
            response_matrix = detrend(response_matrix, axis=-1)
            response_matrix = zscore(response_matrix, axis=-1, nan_policy='omit')

            # split it up into epoch_response_matrix
            # shape = (gloms, trials, timepoints)
            time_vector, epoch_response_matrix = ID.getEpochResponseMatrix(response_matrix, dff=False)
            # Classifier model doesn't like nans
            epoch_response_matrix[np.where(np.isnan(epoch_response_matrix))] = 0

            if classify_on_amplitude:
                classify_data = ID.getResponseAmplitude(epoch_response_matrix, metric='max')[:, :, np.newaxis]
            else:
                classify_data = epoch_response_matrix.copy()
            # output shape = trials x (concatenated glom responses)
            tmp_trials = [classify_data[x, :, :] for x in range(classify_data.shape[0])]
            single_trial_responses = np.concatenate(tmp_trials, axis=-1)

            print('single_trial_responses shape = {}'.format(single_trial_responses.shape))

            cmap = cc.cm.glasbey
            self.colors = cmap(self.included_vals/self.included_vals.max())

            parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
            unique_parameter_values = np.unique(np.array(parameter_values, dtype='object'))
            # Encode param sets to integers in order of unique_parameter_values
            # ref: https://stackoverflow.com/questions/38749305/labelencoder-order-of-fit-for-a-pandas-df
            df = pd.DataFrame(data=np.array(parameter_values, dtype='object'), columns=['params'])
            df['encoded'] = df['params'].apply(lambda x: list(unique_parameter_values).index(x))

            # Filter trials to only include stims of interest
            #   Exclude last 2 (uniform flash)
            #   Exclude one direction of bidirectional stims
            #   Exclude spot on grating for now
            keep_stims = np.array([0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21])
            keep_inds = np.where([x in keep_stims for x in df['encoded'].values])[0]

            included_parameter_values = unique_parameter_values[keep_stims]
            X = single_trial_responses[keep_inds, :]
            y = df['encoded'].values[keep_inds]

            if model_type == 'LogReg':
                classifier_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False)
                # classifier_model = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l1', C=2)
            elif model_type == 'RandomForest':
                classifier_model = RandomForestClassifier(n_estimators=100,
                                                          criterion='gini',
                                                          max_depth=None,
                                                          min_samples_leaf=2,
                                                          random_state=0)

            y_test_all = []
            y_hat_all = []
            for it in range(iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=random_state)

                classifier_model.fit(X_train, y_train)
                y_hat = classifier_model.predict(X_test)

                y_test_all.append(y_test)
                y_hat_all.append(y_hat)

            y_test_all = np.hstack(y_test_all)
            y_hat_all = np.hstack(y_hat_all)
            performance = np.sum(y_hat_all == y_test_all) / y_test_all.shape[0]
            self.overall_performances.append(performance)

            cmat = confusion_matrix(y_test_all, y_hat_all, normalize='true')
            self.cmats.append(cmat)

            if s_ind == pull_eg:  # Pull out some example internal bits for display
                self.unique_parameter_values = unique_parameter_values
                self.included_parameter_values = included_parameter_values
                self.X_test = X_test
                self.y_test = y_test
                self.classifier_model = classifier_model
                tmp_trials = [epoch_response_matrix[x, :, :] for x in range(epoch_response_matrix.shape[0])]
                self.eg_traces = np.concatenate(tmp_trials, axis=-1)
                self.eg_stim_identity = parameter_values

        self.cmats = np.dstack(self.cmats)

        # flies x params
        self.performance = np.vstack([np.diag(self.cmats[:, :, x]) for x in range(self.cmats.shape[-1])])
