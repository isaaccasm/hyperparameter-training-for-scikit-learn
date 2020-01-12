import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

from skopt import forest_minimize


class CVModel(object):
    """
    Implementation of a model that adds a cross validation to a set of scikit-learn models. It does not implement any type of
    objective function. A child should be used instead or an objective function must be passed.
    """

    def __init__(self, models, space, num_iter=50, random_state=5, verbose=False, validation_split=0.1,
                 objective=None):
        """
        Constructor of the CVModel.
        :param models: A list of scikit-learn models
        :param space: A list with the space. Each element in the list is a dictionary for the correspondent element in
                        models. The keys of the dictionary follow the scikit-learn standard to characterise the elements
                        in a model, similar to sklearn.model_selection.GridSearchCV. For instance, if te model name is
                        polynomialfeatures and you want to affect the parameter degree the key would be:
                        'polynomialfeatures__degree'.
                        The values of the dictionary follows the standard of scikit optimize. For instance, a list means
                        specific values [1,2,3], [True, False], ['random', 'type'], a tuple means a set of continuous
                        values like (0,10), any value from 0 to 10.
        :param num_iter: integer Number of iteration. (default 50)
        :param random_state: integer. For the scikit-optimize model. (default 5)
        :param verbose: When True plot some values.
        :param validation_split: The percentage of the data that is going to be used in the validation (default 0.1)
        :param objective: The objective to optimize the model.
        """
        self.num_iter = num_iter
        self.random_state = random_state
        self.verbose = verbose
        self.validation_split = validation_split
        self._id = 0

        self.names = []

        self._models = None
        self.models = models  # if isinstance(models, (list, tuple)) else [models]
        self._model = self._models[0]
        self.space = space if isinstance(space, (list, tuple)) else [space] * len(self._models)

        self.objective = objective

        self.params = [-1, -1]

        self._best_model = None
        self.training_error = -1
        self.all_history = [{'training_loss': [], 'val_loss': [], 'training_acc': [], 'val_acc': []} for _ in
                            range(len(self.models))]
        self.history = self.all_history[self._id]
        self.outputs = {'classifiers': [], 'p_values_error': [], 'p_values_time': []}

    def set_model_parameters(self, params):
        """
        Set the parameters of the model in any scikit learn model. Notice that self.name must have the same convention
        as in the scikit model for accessing parameters of a pipeline model_name+'__'+attribute_name
        :param params: the list of values for the parameters with names in self.names
        :return: None
        """
        # setattr(model.steps[0][1], 'degree', 2)
        attributes = dir(self._model)

        for name, p in zip(self.names, params):
            if hasattr(self._model, 'named_steps'):
                pos = name.find('__')
                if pos < 0:
                    raise NameError(
                        "In a pipeline the name must be model_name+'__'+attribute_name. name {} does not have '__' ".format(
                            name))
                model_name = name[:pos]
                attribute_name = name[pos + 2:]
                attributes = dir(self._model.named_steps[model_name])
                if attribute_name in attributes:
                    setattr(self._model.named_steps[model_name], attribute_name, p)
                else:
                    raise NameError('model {} does not have attribute {}'.format(model_name, attribute_name))
            else:
                if name in attributes:
                    setattr(self._model, name, p)
                else:
                    raise NameError('model {} does not have attribute {}'.format(type(self._model).__name__, name))

    def _objective(self, params, Xt, yt, k=10, return_all=False, **kwargs):
        """
        Objective function for the optimiser. It minimises the root mean squared error of the validation samples.
        It performs a simple k-fold strategy. This method will call the fit method of the model that is being
        cross validated.
        :param params: The parameters passed by the optimiser
        :param Xt: numpy array with the features. Each feature is a column.
        :param yt: numpy array with the labels.
        :param k: int. The number of folds, by default 10.
        :return: The mean error at each fold.
        """
        NotImplementedError('Create a child with an objective function or pass one as input.')

    def welch_test(self, var1, var2):
        """
        Compute the welch test between the results
        :param var1: Values of the first class
        :param var2: Values of the second class
        :return: The p-value
        """
        m1 = np.mean(var1)
        m2 = np.mean(var2)
        s1 = np.std(var1)
        s2 = np.std(var2)
        n1 = len(var1)
        n2 = len(var2)

        vn1 = s1 ** 2 / n1
        vn2 = s2 ** 2 / n2

        st = np.sqrt(vn1 + vn2)

        t = (m1 - m2) / st
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (n1 - 1) + vn2 ** 2 / (n2 - 1))
        df = np.where(np.isnan(df), 1, df)

        return stats.t.sf(np.abs(t), df) * 2

    def fit(self, X, y, test=None, train_only_best=False, **kwargs):
        """
        It performs the fitting of the model. This function uses forest_minimize with an objective function that can
        be either passed when creating the model or use the standard.
        :param X: numpy array with the features. Each feature is a column.
        :param y: numpy array with the labels.
        :param test: A function that compares two lists in order to check whether they are statistically different. They are
                    used to compare the results between models.
        :param train_only_best: When true, only one model is trained, the one selected ins self.id.
        :param kwargs: The parameters of forest_minimize from scikit-optimize
        :return: None
        """

        if train_only_best:
            self._model = self._models[self._id]
        else:
            if test is None:
                test = self.welch_test

            for id in range(len(self.models)):
                if self.verbose:
                    print('Training classifier number {}'.format(id))
                self._fit_single_classifier(id, X, y, **kwargs)

            order = np.argsort([np.mean(out['validation']) for out in self.outputs['classifiers']])
            self._id = order[0]
            self._models = [self._models[ord] for ord in order]
            self.outputs['classifiers'] = [self.outputs['classifiers'][ord] for ord in order]
            self.all_history = [self.all_history[ord] for ord in order]

            n = len(self.outputs['classifiers'])
            p_values_err = np.zeros([n, n])
            p_values_time = np.zeros([n, n])
            for id1, var1 in enumerate(self.outputs['classifiers']):
                for id2, var2 in enumerate(self.outputs['classifiers']):
                    if id1 == id2:
                        p_values_err[id1, id2] = 1.0
                        p_values_time[id1, id2] = 1.0
                    elif id1 > id2:
                        p_values_err[id1, id2] = p_values_err[id2, id1]
                        p_values_time[id1, id2] = p_values_time[id2, id1]
                    else:
                        p_values_err[id1, id2] = test(var1['validation'], var2['validation'])
                        p_values_time[id1, id2] = test(var1['time'], var2['time'])

            self.outputs['p_values_error'] = p_values_err
            self.outputs['p_values_time'] = p_values_time

            self.history = self.all_history[0]
            self._model = self._models[0]
            self.params = self.outputs['classifiers'][0]['best_parameters']

            print('The selected model is {}, the id is: {}'.format(type(self._model).__name__, self._id))

        self.names = list(self.params.keys())
        self.set_model_parameters(list(self.params.values()))
        self._model.fit(X, y)

    def _fit_single_classifier(self, id, X, y, **kwargs):
        """
        DO NOT USE DIRECTLY. Use fit instead. It performs the fitting of a single model.
        :param id: The id of the model to be tested. The method fit handles this.
        :param X: numpy array with the features. Each feature is a column.
        :param y: numpy array with the labels.
        :param kwargs: The parameters of forest_minimize from scikit-optimize
        :return: The
        """

        if self.objective is None:
            self.objective = self._objective

        self._model = self._models[id]
        self.history = self.all_history[id]

        objective = lambda p: self.objective(p, X, y)

        self.names = list(self.space[id].keys())
        res_fm = forest_minimize(objective, list(self.space[id].values()), n_calls=self.num_iter,
                                 random_state=self.random_state,
                                 verbose=self.verbose, **kwargs)

        tr_err, val_err, times = self.objective(res_fm.x, X, y, return_all=True)
        self.outputs['classifiers'].append(
            {'training': tr_err,  # {'mean':np.mean(tr_err), 'std':np.std(tr_err), 'n':len(tr_err), 'samples':tr_err},
             'validation': val_err,
             # {'mean':np.mean(val_err), 'std':np.std(val_err), 'n':len(val_err), 'samples':val_err},
             'time': times,  # {'mean': np.mean(times), 'std': np.std(times), 'n':len(times), 'samples':times},
             'best_parameters': {key: val for key, val in zip(self.space[id], res_fm.x)}})

        self.training_error = self.score(X, y)

    def predict(self, X):
        """
        Performs the prediction of the model.
        :param X: numpy array with the features. Each feature is a column.
        :return: The predicted label.
        """
        return self._model.predict(X)

    def score(self, Xt, yt):
        """
        The mean squared error of the prediction from the features and labels.
        :param Xt: numpy array with the features. Each feature is a column.
        :param yt: numpy array with the labels.
        :return:
        """
        y_val = self.predict(Xt)
        return np.sqrt(mean_squared_error(yt, y_val))

    def test_model(self, Xt, yt):
        """
        It performs a simple testing by comparing the error of the predicted values with always using the most common
        value. This serves to discard a model but it is not very useful for accepting one.
        :param Xt: numpy array with the features. Each feature is a column.
        :param yt: numpy array with the labels.
        :return: None.
        """
        err = self.score(Xt, yt)
        # Assume the prediction is always the most common values
        values, positions = np.histogram(yt, bins=10)
        yt2 = positions[np.argmax(values)]
        base_error = np.sqrt(mean_squared_error(np.array([yt2 for _ in yt]), yt))

        print('Testing error: {}'.format(err))
        print('Training error: {}'.format(self.training_error))
        print('Base error: {}'.format(base_error))

    def plot_error(self, id=0):
        """
        Plot the training and validation error and accuracies
        :return: None
        """
        history = self.all_history[id]

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(history['training_loss'])
        axs[0].plot(history['val_loss'])
        axs[0].legend(['training', 'validation'])
        axs[0].set_title('Training and validation loss')
        axs[0].set_xlabel('Iterations')

        axs[1].plot(history['training_acc'])
        axs[1].plot(history['val_acc'])
        axs[1].legend(['training', 'validation'])
        axs[1].set_title('Training and validation accuracy')
        axs[1].set_xlabel('Iterations')

        plt.show()

    @property
    def model(self):
        """
        :return: The scikit model that has been trained
        """
        if self._model is None:
            print('The selected model is None, run fit before using this method')
        return self._model

    @property
    def models(self):
        """
        Return all the models
        :return:
        """
        return self._models

    @models.setter
    def models(self, models):
        """
        :param models:
        :return:
        """
        if isinstance(models, (tuple, list)):
            self._models = models
        elif hasattr(models, 'fit') and hasattr(models, 'predict') and hasattr(models, 'predict_proba'):
            self._models = [models]
        else:
            ValueError(
                'Models must be tuple or list with classifiers inside or class containing fit, predict and predict_proba')

    @property
    def id(self):
        """
        Return the id of the model that is considered as the best one. Notice that models were organised
        after fit according to their mean validation error and id is always set to 1. This value only changes
        when the user implements something to perform a change.
        :return: The id
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Change the id that represents the best model. Notice that this change is performed after the organisation
        :param id: A single int
        :return: None
        """
        self._id = id
        self._model = self._models[self._id]
        self.history = self.all_history[self._id]
        self.params = self.outputs[self._id]['parameters']

    def save(self, address):
        """
        Save the important parameters of the model.
        :param address: Where to save all the parameters
        :return: None
        """
        data = [self._models, self.params, self.all_history, self._id, self.space]
        with open(address, 'wb') as f:
            pickle.dump(data, f)

    def load(self, address):
        """
        Load the important parameters of the model.
        :param address: Where to save all the parameters
        :return: None
        """
        with open(address, 'rb') as f:
            data = pickle.load(f)

        self._models = data[0]
        self._model = self._models[0]
        self.params = data[1]
        self.all_history = data[2]
        self.history = self.all_history[0]
        self._id = data[3]
        self.space = data[4]