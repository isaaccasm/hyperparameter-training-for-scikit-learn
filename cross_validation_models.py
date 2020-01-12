import time

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.model_selection import KFold

from cross_validation_base import CVModel

class ClassifierCV(CVModel):
    """
    Simple cross validation classifier using log loss
    """

    def __init__(self, *args, kfold=False, num_k=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.kfold = kfold
        self.num_k = num_k

    def _objective_step(self, Xtotal, ytotal, train_indexes, test_indexes):
        """
        :param train_index:
        :param test_index:
        :return:
        """

        X_train, X_val = Xtotal[train_indexes], Xtotal[test_indexes]
        y_train, y_val = ytotal[train_indexes], ytotal[test_indexes]

        self._model.fit(X_train, y_train)
        predict_tr = self._model.predict_proba(X_train)
        err_tr = log_loss(y_train, predict_tr)
        acc_tr = accuracy_score(y_train, self._model.predict(X_train))

        start = time.time()
        predict_val = self._model.predict_proba(X_val)
        time_val = time.time() - start

        err_val = log_loss(y_val, predict_val)
        acc_val = accuracy_score(y_val, self._model.predict(X_val))

        return err_tr, err_val, acc_tr, acc_val, time_val

    def _objective(self, params, Xtotal, ytotal, return_all=False):
        """
        Objective function for the optimiser. It minimises the root mean squared error of the validation samples.
        It performs a simple k-fold strategy. This method will call the fit method of the model that is being
        cross validated.
        :param params: The parameters passed by the optimiser
        :param Xtotal: numpy array with the features. Each feature is a column.
        :param ytotal: numpy array with the labels.
        :param return_all: Returned the training and validation error and time per iteration. This is not used as
                            the standard output because this function is supposed to be used in a scikit-optimise class
                            so that it shoul return a single number representing the error.
        :return: The mean error of all the folds.
        """
        num = len(ytotal)
        self.set_model_parameters(params)
        cpos = int(np.floor(self.validation_split * num))
        validation_error = []
        training_error = []
        validation_acc = []
        training_acc = []
        times = []

        if self.kfold:
            kf = KFold(n_splits=self.num_k, random_state=None, shuffle=True)
            splitted_data = kf.split(Xtotal)
        else:
            indexes = np.arange(len(ytotal))
            splitted_data = []
            for _ in range(self.num_k):
                np.random.shuffle(indexes)
                train_index, test_index = indexes[cpos:], indexes[:cpos]
                splitted_data.append([train_index, test_index])

        for train_indexes, test_indexes in splitted_data:
            err_tr, err_val, acc_tr, acc_val, time = self._objective_step(Xtotal, ytotal, train_indexes, test_indexes)
            training_error.append(err_tr)
            validation_error.append(err_tr)
            training_acc.append(acc_tr)
            validation_acc.append(acc_val)
            times.append(time)

        if self.verbose:
            print('Current error: {}'.format(np.mean(validation_error)))

        self.history['training_loss'].append(np.mean(training_error))
        self.history['val_loss'].append(np.mean(validation_error))
        self.history['training_acc'].append(np.mean(training_acc))
        self.history['val_acc'].append(np.mean(validation_acc))

        if return_all:
            return training_error, validation_error, times

        return np.mean(validation_error)


class RegressorCV(CVModel):
    """
    Simple cross validation regressor, using mean squared error
    """

    def _objective(self, params, Xtotal, ytotal, k=10, return_all=False):
        """
        Objective function for the optimiser. It minimises the root mean squared error of the validation samples.
        It performs a simple k-fold strategy. This method will call the fit method of the model that is being
        cross validated.
        :param params: The parameters passed by the optimiser
        :param Xtotal: numpy array with the features. Each feature is a column.
        :param ytotal: numpy array with the labels.
        :param k: int. The number of folds, by default 10.
        :param return_all: Returned the training and validation error and time per iteration. This is not used as
                            the standard output because this function is supposed to be used in a scikit-optimise class
                            so that it should return a single number representing the error.
        :return: The mean error at each fold.
        """
        num = len(ytotal)
        self.set_model_parameters(params)
        cpos = int(np.floor(self.validation_split * num))
        validation_error = []
        training_error = []
        times = []
        for _ in range(k):
            # Split into validation and training. THIS IS THE SIMPLEST CROSS-VALIDATION. However, it can be repeated
            # several times because it is fast.
            pos = np.arange(len(ytotal))
            np.random.shuffle(pos)

            self._model.fit(Xtotal[pos[cpos:]], ytotal[pos[cpos:]])
            values = self._model.predict(Xtotal[pos[:cpos]])

            # training and testing errors
            err = mean_squared_error(ytotal[pos[:cpos]], values)
            validation_error.append(err)

            self._model.fit(Xtotal[pos[cpos:]], ytotal[pos[cpos:]])
            start = time.time()
            values = self._model.predict(Xtotal[pos[cpos:]])
            times.append(time.time() - start)

            training_error.append(mean_squared_error(ytotal[pos[cpos:]], values))

        if self.verbose:
            print('Current error: {}'.format(np.mean(validation_error)))

        self.history['training_loss'].append(np.mean(training_error))
        self.history['val_loss'].append(np.mean(validation_error))

        if return_all:
            return training_error, validation_error, times

        return np.mean(validation_error)
