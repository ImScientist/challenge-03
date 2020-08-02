import numpy as np


class GBMWrapped(object):
    """ A wrapper of a trained binary classification gbm model
    such that it can be processed to the sklearn.calibration library
    """
    def __init__(self, predictor):
        self.predictor = predictor
        self.classes_ = [0, 1]

    def predict(self, X):
        return self.predictor.predict(X)

    def predict_proba(self, X):
        res = self.predictor.predict(X).reshape(-1, 1)
        return np.concatenate(((1 - res), res), axis=1)
