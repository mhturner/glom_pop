"""

"""
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import colorcet as cc

from visanalysis.analysis import volumetric_data
from glom_pop import dataio


class SingleTrialEncoding():
    def __init__(self):
        self.experiment_file_directory = '/Users/mhturner/CurrentData'

    def evaluatePerformance(self, dataset, included_gloms, iterations=20, pull_eg=0):

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
            erm = []
            included_vals = []
            for g_ind, name in enumerate(included_gloms):
                pull_ind = np.where(name == names)[0][0]
                erm.append(response_data.get('epoch_response')[pull_ind, :, :])
                included_vals.append(vals[pull_ind])
            epoch_response_matrix = np.stack(erm, axis=0)
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
            # shape = trials x time (concatenated glom responses)
            single_trial_responses = np.reshape(epoch_response_matrix, (-1, epoch_response_matrix.shape[2])).T

            # Filter trials to only include stims of interest
            # exclude last 2 (uniform flash)
            # Exclude one direction of bidirectional stims
            # Exclude spot on grating for now
            keep_stims = np.array([0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21])
            keep_inds = np.where([x in keep_stims for x in df['encoded'].values])[0]

            included_parameter_values = unique_parameter_values[keep_stims]
            X = single_trial_responses[keep_inds, :]
            y = df['encoded'].values[keep_inds]

            LogRegModel = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False)
            # LogRegModel = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l1', C=2)

            y_test_all = []
            y_hat_all = []
            for it in range(iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

                LogRegModel.fit(X_train, y_train)
                y_hat = LogRegModel.predict(X_test)

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
                self.LogRegModel = LogRegModel

        self.cmats = np.dstack(self.cmats)

        # flies x params
        self.performance = np.vstack([np.diag(self.cmats[:, :, x]) for x in range(self.cmats.shape[-1])])
