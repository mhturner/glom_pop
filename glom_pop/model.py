"""

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.signal import detrend
from scipy.stats import zscore

from visanalysis.analysis import imaging_data
from glom_pop import dataio, util


class SingleTrialEncoding_onefly():
    def __init__(self, ID, included_gloms):
        self.ID = ID
        self.included_gloms = included_gloms
        self.included_vals = dataio.get_glom_vals_from_names(included_gloms)

    def prep_model(self):
        # Load response data
        response_data = dataio.load_responses(self.ID, response_set_name='glom', get_voxel_responses=False)

        # Only select gloms in included_gloms
        glom_size_threshold = 10
        # response_matrix: shape=(gloms, time)
        response_matrix = np.zeros((len(self.included_gloms), response_data.get('response').shape[1]))
        for glom_ind, included_glom in enumerate(self.included_gloms):
            new_glom_size = np.sum(response_data.get('mask') == self.included_vals[glom_ind])

            if new_glom_size > glom_size_threshold:
                pull_ind = np.where(self.included_vals[glom_ind] == response_data.get('mask_vals'))[0][0]
                response_matrix[glom_ind, :] = response_data.get('response')[pull_ind, :]
            else:  # Exclude because this glom, in this fly, is too tiny
                pass
        response_matrix = np.stack(response_matrix, axis=0)

        # Detrend (remove ~linear bleach) & z-score
        response_matrix = detrend(response_matrix, axis=-1)
        response_matrix = zscore(response_matrix, axis=-1, nan_policy='omit')

        # split it up into epoch_response_matrix
        # shape = (gloms, trials, timepoints)
        time_vector, epoch_response_matrix = self.ID.getEpochResponseMatrix(response_matrix, dff=False)
        # Classifier model doesn't like nans
        epoch_response_matrix[np.where(np.isnan(epoch_response_matrix))] = 0

        # classify_on_amplitude:
        classify_data = self.ID.getResponseAmplitude(epoch_response_matrix, metric='max')[:, :, np.newaxis]

        parameter_values = [list(pd.values()) for pd in self.ID.getEpochParameterDicts()]
        self.unique_parameter_values = np.unique(np.array(parameter_values, dtype='object'))
        # Encode param sets to integers in order of unique_parameter_values
        # ref: https://stackoverflow.com/questions/38749305/labelencoder-order-of-fit-for-a-pandas-df
        df = pd.DataFrame(data=np.array(parameter_values, dtype='object'), columns=['params'])

        # x: response matrix
        #    shape = trials x (concatenated glom responses)
        tmp_trials = [classify_data[x, :, :] for x in range(classify_data.shape[0])]
        self.x = np.concatenate(tmp_trials, axis=-1)

        # y: stim encoding for each trial
        self.y = df['params'].apply(lambda x: list(self.unique_parameter_values).index(x)).values

        self.classifier_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False)

    def train_test_model(self, train_inds, test_inds):
        self.classifier_model.fit(self.x[train_inds, :],
                                  self.y[train_inds])
        self.y_hat = self.classifier_model.predict(self.x[test_inds, :])
        self.y_true = self.y[test_inds]

        self.performance = np.sum(self.y_hat == self.y_true) / self.y_true.shape[0]
        self.cmat = confusion_matrix(self.y_true, self.y_hat, normalize='true')


class SingleTrialEncoding():
    def __init__(self, data_series, included_gloms, experiment_file_directory='/Users/mhturner/CurrentData'):
        self.experiment_file_directory = experiment_file_directory
        self.data_series = data_series
        self.included_gloms = included_gloms
        self.included_vals = dataio.get_glom_vals_from_names(included_gloms)

    def evaluate_performance(self, model_type='LogReg',
                             iterations=20, pull_eg=0,
                             classify_on_amplitude=True, random_state=None,
                             shuffle_trials=False,
                             stim_set=None):

        self.cmats = []
        self.overall_performances = []
        if stim_set is None:
            self.stim_set = 'PGS'
        else:
            self.stim_set = stim_set

        for s_ind, series in enumerate(self.data_series):
            series_number = series['series']
            file_path = series['file_name'] + '.hdf5'
            ID = imaging_data.ImagingDataObject(file_path,
                                                series_number,
                                                quiet=True)

            # Load response data
            response_data = dataio.load_responses(ID, response_set_name='glom', get_voxel_responses=False)

            # Only select gloms in included_gloms
            glom_size_threshold = 10
            # response_matrix: shape=(gloms, time)
            response_matrix = np.zeros((len(self.included_gloms), response_data.get('response').shape[1]))
            for glom_ind, included_glom in enumerate(self.included_gloms):
                new_glom_size = np.sum(response_data.get('mask') == self.included_vals[glom_ind])

                if new_glom_size > glom_size_threshold:
                    pull_ind = np.where(self.included_vals[glom_ind] == response_data.get('mask_vals'))[0][0]
                    response_matrix[glom_ind, :] = response_data.get('response')[pull_ind, :]
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

            # print('single_trial_responses shape = {}'.format(single_trial_responses.shape))

            self.colors = [util.get_color_dict()[x] for x in self.included_gloms]

            parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
            unique_parameter_values = np.unique(np.array(parameter_values, dtype='object'))
            # Encode param sets to integers in order of unique_parameter_values
            # ref: https://stackoverflow.com/questions/38749305/labelencoder-order-of-fit-for-a-pandas-df
            df = pd.DataFrame(data=np.array(parameter_values, dtype='object'), columns=['params'])
            df['encoded'] = df['params'].apply(lambda x: list(unique_parameter_values).index(x))

            if self.stim_set == 'PGS':
                # Filter trials to only include stims of interest
                #   Exclude last 2 (uniform flash)
                #   Exclude one direction of bidirectional stims
                #   Exclude spot on grating
                keep_stims = np.array([0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21])
                keep_inds = np.where([x in keep_stims for x in df['encoded'].values])[0]

                included_parameter_values = unique_parameter_values[keep_stims]
                X = single_trial_responses[keep_inds, :]
                y = df['encoded'].values[keep_inds]
            else:  # Decode all stims presented
                pass
                included_parameter_values = unique_parameter_values
                X = single_trial_responses
                y = df['encoded'].values

            if model_type == 'LogReg':
                classifier_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False)
            elif model_type == 'RandomForest':
                classifier_model = RandomForestClassifier(n_estimators=100,
                                                          criterion='gini',
                                                          max_depth=None,
                                                          min_samples_leaf=2,
                                                          random_state=0)

            y_test_all = []
            y_hat_all = []
            for it in range(iterations):
                if shuffle_trials:
                    assert classify_on_amplitude, 'shuffle_trials only implemented for amplitude classification'
                    for stim_ind, stim in enumerate(np.unique(y)):
                        matching_inds = np.where(y == stim)[0]
                        subset = X[matching_inds, :]
                        c_pre = np.nanmean(np.corrcoef(subset.T))

                        idx = np.random.rand(*subset.shape).argsort(0)
                        shuffled = subset[idx, np.arange(subset.shape[1])]
                        X[matching_inds, :] = shuffled
                        c_shuffle = np.nanmean(np.corrcoef(shuffled.T))

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
