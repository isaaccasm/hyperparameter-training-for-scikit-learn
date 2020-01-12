# hyperparameter-training-for-scikit-learn
Framework using scikit-optimise to train hyperparameters for scikit-learn models

There are three files:

## Base class
The file `cross_validation_base.py` contains the base class. This class should be enough to perform the cross 
validation of any model. However, this class does not implement the objective function since this one must be tailored
to the application. The objective must specified the los and the type of cross validation like k fold.
There are two ways to pass the objective.

### Inheritance
Create a child from `CVModel` and implement the method: `_objective`. This is the recommended way and there are 
two examples of this in the file `cross_validation_models.py`

### As an input
When creating the object from the class `CVModel` pass it as an input.
``` 
cv_model = CVModel(*args, objectve=my_objective)
```

## Children classes
The file `cross_validation_models` contains two example of classes inheriting from `CVModel` one using the logloss 
(`ClassifierCV`) an another one with mean squared error (`RegressorCV`). These two classes can be used for other classes,
but the objective are quite simple and only the KFold cross validation is implemented.

## Examples
This file contains an example of `ClassifierCV` and another of `RegresorCV`.