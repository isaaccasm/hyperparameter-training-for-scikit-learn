from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cross_validation_models import ClassifierCV

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



if __name__ == '__main__':
    classifier_example()