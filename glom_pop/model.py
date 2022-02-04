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

from visanalysis.analysis import volumetric_data
from glom_pop import dataio


class SingleTrialEncoding():
    def __init__(self):
        self.experiment_file_directory = '/Users/mhturner/CurrentData'

    def evaluatePerformance(self, dataset, included_gloms, iterations=20, pull_eg=0, model_type='LogReg'):

        self.cmats = []
        self.overall_performances = []
        for s_ind, key in enumerate(dataset):
            experiment_file_name = key.split('_')[0]
            series_number = int(key.split('_')[1])

            file_path = os.path.join(self.experiment_file_directory, experiment_file_name + '.hdf5')
            ID = volumetric_data.VolumetricDataObject(file_path,
                                                      series_number,
                                                      quiet=True)

            # Load response data
            response_data = dataio.loadResponses(ID, response_set_name='glom', get_voxel_responses=False)
            vals, names = dataio.getGlomMaskDecoder(response_data.get('mask'))

            # Only select gloms in included_gloms
            response_matrix = []
            included_vals = []
            for g_ind, name in enumerate(included_gloms):
                pull_ind = np.where(name == names)[0][0]
                response_matrix.append(response_data.get('response')[pull_ind, :])
                included_vals.append(vals[pull_ind])
            response_matrix = np.stack(response_matrix, axis=0)

            # Detrend (remove linear bleach) & z-score
            response_matrix = detrend(response_matrix, axis=-1)
            response_matrix = zscore(response_matrix, axis=-1)

            # NaN comes from empty gloms (this fly doesn't have that glom)
            if np.any(np.isnan(response_matrix)):
                nan_row = np.where(np.any(np.isnan(response_matrix), axis=1))[0]
                response_matrix[nan_row, :] = 0
                print('{} {}:Setting NaN to 0 in row(s) {}'.format(s_ind, key, nan_row))

            # split it up into epoch_response_matrix
            # shape = (gloms, trials, timepoints)
            time_vector, epoch_response_matrix = ID.getEpochResponseMatrix(response_matrix, dff=False)

            included_vals = np.array(included_vals)
            cmap = cc.cm.glasbey
            self.colors = cmap(included_vals/included_vals.max())

            parameter_values = [list(pd.values()) for pd in ID.getEpochParameterDicts()]
            unique_parameter_values = np.unique(np.array(parameter_values, dtype='object'))
            # Encode param sets to integers in order of unique_parameter_values
            # ref: https://stackoverflow.com/questions/38749305/labelencoder-order-of-fit-for-a-pandas-df
            df = pd.DataFrame(data=np.array(parameter_values, dtype='object'), columns=['params'])
            df['encoded'] = df['params'].apply(lambda x: list(unique_parameter_values).index(x))

            # Multinomial logistic regression model
            # output shape = trials x time (concatenated glom responses)
            tmp_trials = [epoch_response_matrix[x, :, :] for x in range(epoch_response_matrix.shape[0])]
            single_trial_responses = np.concatenate(tmp_trials, axis=-1)

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
                classifier_model = RandomForestClassifier(max_depth=2, random_state=0)

            y_test_all = []
            y_hat_all = []
            for it in range(iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

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

        self.cmats = np.dstack(self.cmats)

        # flies x params
        self.performance = np.vstack([np.diag(self.cmats[:, :, x]) for x in range(self.cmats.shape[-1])])
