__author__ = 'mikhail91'

import numpy
import pandas
from sklearn.linear_model import LinearRegression
import itertools

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
        self.robust_lr = None
        self.robust_score = None

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

        score = numpy.sqrt(1. * (numpy.abs(lr.coef_[0,0] * X_new + lr.intercept_[0] - Y_new)**2).sum() / len(X_new))

        return score, lr

    def score(self, X, Rx, Ry):
        pass


    def predict(self, X):
        """
        Predict Ys.
        ----
        :param numby.ndarray X: array of Xs of the wires centers with shape (n_samples, 1).

        :return numby.ndarray with shape (n_samples, 1)
        """
        return self.regressor.predict(X)


class RobustMarginLinearRegression(object):

    def __init__(self, n_iter=10):

        self.n_iter = n_iter
        self.robust_lr = None
        self.robust_score = 100000

    def fit(self, hits):

        StatNbs = numpy.unique(hits.StatNb.values)
        ViewNb = numpy.unique(hits.ViewNb.values)

        Combs = numpy.unique(hits.StatNb.values*10 + hits.ViewNb.values)

        ind_sets = []
        for comb in Combs:
            one_set = hits[(hits.StatNb == comb//10) * (hits.ViewNb == comb%10)].index
            ind_sets.append(one_set)

        indexes_list = list(itertools.product(*ind_sets))

        for ind in indexes_list:

            ind = list(ind)

            Wz = hits.loc[ind].Wz.values.reshape(-1, 1)
            Wy = hits.loc[ind].Wy.values.reshape(-1, 1)
            R = hits.loc[ind].dist2Wire.values.reshape(-1, 1)

            mlr = MarginLinearRegression(n_iter=self.n_iter)
            score, lr = mlr.fit(Wz, Wy, R, R)

            if score < self.robust_score:
                self.robust_score = score
                self.robust_lr = lr

        return self.robust_score, self.robust_lr

    def dist2track(self, hits):

        Wz = hits.Wz.values.reshape(-1, 1)
        Wy = hits.Wy.values.reshape(-1, 1)
        R = hits.dist2Wire.values.reshape(-1, 1)

        k = self.robust_lr.coef_[0,0]
        b = self.robust_lr.intercept_[0]

        Sign = (1. * (Wy > k * Wz + b) - 0.5) * 2.
        Wz_new = Wz + Sign * R * k/numpy.sqrt(1 + k**2)
        Wy_new = Wy - Sign * R * 1./numpy.sqrt(1 + k**2)

        dists = numpy.abs(k * Wz_new + b - Wy_new)

        return dists

    def turbo_fit(self, hits, clf):

        score, lr = self.fit(hits)

        dist = self.dist2track(hits)[:, 0]

        data = pandas.DataFrame()
        data['dist'] = dist
        data['score'] = [score]*len(dist)
        data['score_div_dist'] = data['score'].values / (data['dist'].values + 0.000001)
        data['slope'] = [lr.coef_[0,0]]*len(dist)
        data['intercept'] = [lr.intercept_[0]]*len(dist)

        prediction = clf.predict(data)

        track = hits[prediction == 1]

        if len(track) >= 3:

            Wz = track.Wz.values.reshape(-1, 1)
            Wy = track.Wy.values.reshape(-1, 1)
            R = track.dist2Wire.values.reshape(-1, 1)

            mlr = MarginLinearRegression(n_iter=self.n_iter)
            score, lr = mlr.fit(Wz, Wy, R, R)

        return score, lr, track





