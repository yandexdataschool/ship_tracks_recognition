__author__ = 'mikhail91'

import numpy
from sklearn.linear_model import LinearRegression

class MarginLinearRegression(object):
    """
    This is Margin Linear Regression which is used distances to the wires to fit
    linear regression.
    ----
    :param int n_iter: number of iterations of the EM algorithm.
    """
    def __init__(self, n_iter=10):

        self.slope = None # slope of the line
        self.intersept = None # intersept of the line
        self.regressor = None # regressor's class
        self.n_iter = n_iter # number of iterations
        self.hits_X = None # Xs of the hits, which are reconstructed from the regression
        self.hits_Y = None # Ys of the hits, which are reconstructed from the regression

    def fit(self, X, Y, Rx, Ry):
        """
        Fit the Margin Linear Regresiion.
        ----
        :param numby.ndarray X: array of Xs of the wires centers with shape (n_samples, 1).

        :param numby.ndarray Y: array of Ys of the wires centers with shape (n_samples, 1).

        :param numby.ndarray Rx: array of distances on X axis to the wires centers with shape (n_samples, 1).

        :param numby.ndarray Ry: array of distances on Y axis to the wires centers with shape (n_samples, 1).
        """
        lr = LinearRegression()
        weights = (numpy.sqrt((1./Rx)**2 + (1./Ry)**2)).reshape(len(X), )
        lr.fit(X, Y, weights)
        slope = lr.coef_[0,0]
        Sign = (1. * (Y > lr.predict(X)) - 0.5) * 2.

        for iter in range(0, self.n_iter):
            X_new = X + Sign * Rx * slope/numpy.sqrt(1 + slope**2)
            Y_new = Y - Sign * Ry * 1./numpy.sqrt(1 + slope**2)
            lr.fit(X_new, Y_new)#, weights)
            slope = lr.coef_[0,0]
            Sign = (1. * (Y > lr.predict(X)) - 0.5) * 2.

        self.regressor = lr
        self.hits_X = X_new
        self.hits_Y = Y_new

    def predict(self, X):
        """
        Predict Ys.
        ----
        :param numby.ndarray X: array of Xs of the wires centers with shape (n_samples, 1).

        :return numby.ndarray with shape (n_samples, 1)
        """
        return self.regressor.predict(X)