import numpy as np

from sklearn import datasets
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cross_validation_models import ClassifierCV, RegressorCV

def classifier_example():
    iris = datasets.load_iris()
    X = iris.data[:, 0:2]  # we only take the first two features for visualization
    y = iris.target

    rf = RandomForestClassifier()
    sp_rf = {'min_samples_leaf': list(range(1,30)), 'min_weight_fraction_leaf':(0, 0.4), 'bootstrap':[True, False]}

    lr = make_pipeline(PolynomialFeatures(), LogisticRegression())
    sp_lr = {'polynomialfeatures__degree':[1,2,3], 'logisticregression__C':(0.01, 10), 'logisticregression__tol':(0.001,0.1)}

    svm = SVC(kernel='linear', probability=True)
    sp_svm = {'C': (0.01, 10)}

    svm_rbf = SVC(kernel='rbf', probability=True)
    sp_svm_rbf = {'C': (0.01, 10), 'gamma':[0.01, 10]}

    space = [sp_rf, sp_lr, sp_svm, sp_svm_rbf]
    models = [rf, lr, svm, svm_rbf]

    clf = ClassifierCV(models, space, validation_split=0.15, verbose=True, num_iter=11)

    clf.fit(X, y)

    print(clf.outputs)
    clf.plot_error()

def regressor_example():
    # Generate sample data
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    rf = RandomForestRegressor()
    sp_rf = {'n_estimators':list(range(10,200)),'min_samples_leaf': list(range(1, 30)), 'min_weight_fraction_leaf': (0, 0.4)}

    svr_rbf = SVR(kernel='rbf', epsilon=.1)
    sp_svr_rbf = {'C': np.logspace(0.1, 1000, num=500), 'gamma': [0.01, 1]}

    svr_lin = SVR(kernel='linear', gamma='auto')
    sp_svr_lin = {'C': np.logspace(0.1, 500, num=500)}

    svr_poly = SVR(kernel='poly', epsilon=.1)
    sp_svr_poly = {'C': np.logspace(0.1, 1000, num=500), 'gamma': [0.01, 1], 'degree':[2,3], 'coef0':(-5,5)}

    kr = KernelRidge(kernel='rbf'),
    sp_kr = {"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}

    space = [sp_rf, sp_kr, sp_svr_lin, sp_svr_rbf, sp_svr_poly]
    models = [rf, kr, svr_lin, svr_rbf, svr_poly]

    clf = RegressorCV(models, space, validation_split=0.15, verbose=True, num_iter=11)
    clf.fit(X, y)

    print(clf.outputs)
    clf.plot_error()

if __name__ == '__main__':
    regressor_example()