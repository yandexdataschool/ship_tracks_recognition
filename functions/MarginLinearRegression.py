__author__ = 'mikhail91'

import numpy
import pandas
from sklearn.linear_model import LinearRegression
import itertools
import UV_views

from numpy.linalg import inv

class QuasiLinearRegression(object):

    def __init__(self):
        self.k = None


    def fit(self, X, y, weights=None, a=0):

        I_arr = numpy.ones((len(X), 1))
        X_arr = X
        Z_arr = X_arr**2
        X = numpy.concatenate((I_arr, X_arr, Z_arr), axis=1)

        X = numpy.matrix(X)
        Z = numpy.matrix(Z_arr)
        y = numpy.matrix(y)

        if weights != None:
            W = numpy.diag(weights.reshape((-1,)))
        else:
            E = numpy.eye(len(y))
            W = numpy.matrix(E)

        k = inv(X.T * W * X) * (X.T * W * y)
        err = (y - X * k).T * W * (y - X * k)

        self.k = k

        if numpy.abs(k[2, 0]) > a:

            X = numpy.concatenate((I_arr, X_arr), axis=1)
            X = numpy.matrix(X)

            k_p = inv(X.T * W * X) * (X.T * W * (y - Z * a))
            k_p_new = numpy.concatenate((k_p, [[a]]))
            err_p = (y - X * k_p - Z * a).T * W * (y - X * k_p - Z * a)

            k_m = inv(X.T * W * X) * (X.T * W * (y + Z * a))
            k_m_new = numpy.concatenate((k_m, [[-a]]))
            err_m = (y - X * k_m + Z * a).T * W * (y - X * k_m + Z * a)

            if err_m < err_p:
                self.k = k_m_new
                #return k_m_new
            else:
                self.k = k_p_new
                #return k_p_new

        else:
            #return k
            pass

    def predict(self, X):

        I_arr = numpy.ones((len(X), 1))
        X_arr = X
        Z_arr = X_arr**2
        X = numpy.concatenate((I_arr, X_arr, Z_arr), axis=1)

        X = numpy.matrix(X)

        k = self.k

        y = X * k

        return numpy.array(y)

    def slope(self, X):

        k = self.k

        slope = k[1,0] + 2. * k[2, 0] * X

        return slope




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
        qlr = LinearRegression()
        weights = (numpy.sqrt((1./Rx)**2 + (1./Ry)**2)).reshape(len(X), )
        qlr.fit(X, Y, weights)
        slope = qlr.coef_[0]
        Sign = (1. * (Y > qlr.predict(X)) - 0.5) * 2.

        for iter in range(0, self.n_iter):
            X_new = X + Sign * Rx * slope/numpy.sqrt(1 + slope**2)
            Y_new = Y - Sign * Ry * 1./numpy.sqrt(1 + slope**2)
            qlr.fit(X_new, Y_new)#, weights)
            slope = qlr.coef_[0]
            Sign = (1. * (Y > qlr.predict(X)) - 0.5) * 2.

        self.regressor = qlr
        self.hits_X = X_new
        self.hits_Y = Y_new

        if len(X_new) > 2:
            score = numpy.sqrt(1. * (numpy.abs(qlr.predict(X_new) - Y_new)**2).sum() / (len(X_new) - 2))
        else:
            score = numpy.sqrt(1. * (numpy.abs(qlr.predict(X_new) - Y_new)**2).sum() / (len(X_new) - 1))

        return score, qlr

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

    def __init__(self, n_iter=10, view='Y'):

        self.n_iter = n_iter
        self.robust_lr = None
        self.robust_score = 100000
        self.view = view



    def fit(self, hits):

        StatNbs = numpy.unique(hits.StatNb.values)
        ViewNb = numpy.unique(hits.ViewNb.values)

        Combs = numpy.unique(hits.StatNb.values * 10 + hits.ViewNb.values)

        ind_sets = []

        for comb in Combs:

            one_set = hits[(hits.StatNb.values == comb // 10) * (hits.ViewNb.values == comb % 10)].index
            ind_sets.append(one_set)

        indexes_list = list(itertools.product(*ind_sets))

        for ind in indexes_list:

            ind = list(ind)

            if self.view == 'Y':

                X = hits.loc[ind].Wz.values.reshape(-1, 1)
                Y = hits.loc[ind].Wy.values.reshape(-1, 1)

                Rx = hits.loc[ind].dist2Wire.values.reshape(-1, 1)
                Ry = hits.loc[ind].dist2Wire.values.reshape(-1, 1)

            if self.view == 'Stereo':

                X = hits.loc[ind].Wz.values.reshape(-1, 1)
                Y = hits.loc[ind].Wx.values.reshape(-1, 1)

                Rx = hits.loc[ind].dist2Wire.values.reshape(-1, 1)
                Ry = hits.loc[ind].dist2Wire.values.reshape(-1, 1) / numpy.sin(5. * numpy.pi / 180.)

            mlr = MarginLinearRegression(n_iter=self.n_iter)
            score, lr = mlr.fit(X, Y, Rx, Ry)

            if score < self.robust_score:
                self.robust_score = score
                self.robust_lr = lr

        return self.robust_score, self.robust_lr

    def dist2track(self, hits):

        if self.view == 'Y':

            X = hits.Wz.values.reshape(-1, 1)
            Y = hits.Wy.values.reshape(-1, 1)

            Rx = hits.dist2Wire.values.reshape(-1, 1)
            Ry = hits.dist2Wire.values.reshape(-1, 1)

        if self.view == 'Stereo':

            X = hits.Wz.values.reshape(-1, 1)
            Y = hits.Wx.values.reshape(-1, 1)

            Rx = hits.dist2Wire.values.reshape(-1, 1)
            Ry = hits.dist2Wire.values.reshape(-1, 1) / numpy.sin(5. * numpy.pi / 180.)

        k = self.robust_lr.coef_[0]
        #b = self.robust_lr.intercept_[0]

        Sign = (1. * (Y > self.robust_lr.predict(X)) - 0.5) * 2.
        X_new = X + Sign * Rx * k/numpy.sqrt(1 + k**2)
        Y_new = Y - Sign * Ry * 1./numpy.sqrt(1 + k**2)

        dists = numpy.abs(self.robust_lr.predict(X_new) - Y_new)

        return dists

    def get_zx_hits(self, event, zy_plane_k, zy_plane_b):

        stereo_hits = UV_views.modify_for_xz_analysis_1_2(event)

        stereo_hits['Wy'] = zy_plane_k * stereo_hits.Wz.values + zy_plane_b
        stereo_hits['Wx'] = (stereo_hits.Wx2.values - stereo_hits.Wx1.values) / (stereo_hits.Wy2.values - stereo_hits.Wy1.values) * \
                            (stereo_hits.Wy.values - stereo_hits.Wy1.values) + stereo_hits.Wx1.values

        return stereo_hits[numpy.abs(stereo_hits.Wx.values) < 300]

    def turbo_fit(self, hits, clf, event):

        score, lr = self.fit(hits)

        dist = self.dist2track(hits)[:, 0]

        data = pandas.DataFrame()
        data['dist'] = dist
        data['all_dist2wire'] = hits.dist2Wire.values
        data['score'] = [score]*len(dist)
        data['score_div_dist'] = data['score'].values / (data['dist'].values + 0.000001)
        data['slope'] = [lr.coef_[0]]*len(dist)
        data['intercept'] = [lr.intercept_]*len(dist)
        data['f1'] = data['score'].values - data['dist'].values
        data['f2'] = data['dist'].values/data['all_dist2wire'].values

        stereo_hits = self.get_zx_hits(event, lr.coef_[0], lr.intercept_)
        rmlr_stereo = RobustMarginLinearRegression(2, view='Stereo')
        #print stereo_hits
        if len(stereo_hits) > 4:
            score_stereo, lr_stereo = rmlr_stereo.fit(stereo_hits)
            #dist_stereo = rmlr_stereo.dist2track(stereo_hits).reshape((-1,))
            #dist_stereo.sort()
            #dist_stereo = dist_stereo[:4].mean()
        else:
            dist_stereo = 1000
            score_stereo = 1000

        data['all_dist2wires_stereo'] = [score_stereo] * len(hits)



        #print data

        #train_cols = ['dist', 'all_dist2wire', 'score', 'score_div_dist', 'slope', 'intercept', 'k2', 'f1', 'f2']
        train_cols = [u'dist', u'all_dist2wire', 'all_dist2wires_stereo', u'score', u'score_div_dist']

        prediction = clf.predict(data[train_cols].values)
        #prediction = clf.predict(data.values)

        #print data, prediction
        #print prediction

        track = hits[prediction == 1]

        if len(track) >= 3:

            if self.view == 'Y':

                X = track.Wz.values.reshape(-1, 1)
                Y = track.Wy.values.reshape(-1, 1)

                Rx = track.dist2Wire.values.reshape(-1, 1)
                Ry = track.dist2Wire.values.reshape(-1, 1)

            if self.view == 'Stereo':

                X = track.Wz.values.reshape(-1, 1)
                Y = track.Wx.values.reshape(-1, 1)

                Rx = track.dist2Wire.values.reshape(-1, 1)
                Ry = track.dist2Wire.values.reshape(-1, 1) / numpy.sin(5. * numpy.pi / 180.)

            mlr = MarginLinearRegression(n_iter=self.n_iter)
            score, lr = mlr.fit(X, Y, Rx, Ry)

        return score, lr, track



import UV_views

def get_zx_hits(event, zy_plane_k, zy_plane_b):

    stereo_hits = UV_views.modify_for_xz_analysis_1_2(event)

    stereo_hits['Wy'] = zy_plane_k * stereo_hits.Wz.values + zy_plane_b
    stereo_hits['Wx'] = (stereo_hits.Wx2.values - stereo_hits.Wx1.values) / (stereo_hits.Wy2.values - stereo_hits.Wy1.values) * \
                        (stereo_hits.Wy.values - stereo_hits.Wy1.values) + stereo_hits.Wx1.values

    return stereo_hits#[numpy.abs(stereo_hits.Wx.values) < 300]

# def turbo_fit(track, event, clf_y, clf_stereo):
#
#     # y views
#     rmlr_y = RobustMarginLinearRegression(2)
#     score_y, lr_y = rmlr_y.fit(track)
#
#     dists_y = rmlr_y.dist2track(track)[:, 0]
#
#     data_y = pandas.DataFrame()
#     data_y['dist'] = dists_y
#     data_y['dist2wire'] = track.dist2Wire.values
#     data_y['score'] = [score_y] * len(track)
#     data_y['slope'] = [lr_y.coef_[0][0]] * len(track)
#     data_y['intercept'] = [lr_y.intercept_[0]] * len(track)
#     data_y['dist_div_dist2wire'] = data_y['dist'].values / data_y['dist2wire'].values
#
#     #all_cols = [u'dist', u'dist2wire', u'score', u'dist_div_dist2wire', 'slope', 'intercept']
#     all_cols = [u'dist', u'dist2wire', u'score', u'dist_div_dist2wire', 'slope', 'intercept']
#
#     predict_y = clf_y.predict(data_y[all_cols].values)
#
#     # stereo views
#     stereo_hits = get_zx_hits(event, lr_y.coef_[0][0], lr_y.intercept_[0])
#
#     rmlr_stereo = RobustMarginLinearRegression(2, view='Stereo')
#     score_stereo, lr_stereo = rmlr_stereo.fit(stereo_hits)
#
#     dists_stereo = rmlr_stereo.dist2track(stereo_hits)[:, 0]
#
#     data_stereo = pandas.DataFrame()
#     data_stereo['dist'] = dists_stereo
#     data_stereo['dist2wire'] = stereo_hits.dist2Wire.values
#     data_stereo['score'] = [score_stereo] * len(stereo_hits)
#     data_stereo['slope'] = [lr_stereo.coef_[0][0]] * len(stereo_hits)
#     data_stereo['intercept'] = [lr_stereo.intercept_[0]] * len(stereo_hits)
#     data_stereo['dist_div_dist2wire'] = data_stereo['dist'].values / data_stereo['dist2wire'].values
#
#     predict_stereo = clf_stereo.predict(data_stereo[all_cols].values)
#
#     if (predict_y == 1).sum() >= 4 or (predict_stereo == 1).sum() >= 4:
#
#         return score_y, lr_y, track[predict_y != 0]
#
#     else:
#
#         return score_y, lr_y, track[predict_y == 1]

def turbo_fit(track, event, clf, view):

    # y views
    rmlr = RobustMarginLinearRegression(2, view=view)
    score, lr = rmlr.fit(track)

    dists = rmlr.dist2track(track)[:, 0]

    data = pandas.DataFrame()
    data['dist'] = dists
    data['dist2wire'] = track.dist2Wire.values
    data['score'] = [score] * len(track)
    data['slope'] = [lr.coef_[0][0]] * len(track)
    data['intercept'] = [lr.intercept_[0]] * len(track)
    data['dist_div_dist2wire'] = data['dist'].values / data['dist2wire'].values

    #all_cols = [u'dist', u'dist2wire', u'score', u'dist_div_dist2wire', 'slope', 'intercept']
    all_cols = [u'dist', u'dist2wire', u'score', u'dist_div_dist2wire', 'slope', 'intercept']

    predict_y = clf.predict(data[all_cols].values)

    proba = clf.predict_proba(data[all_cols].values)

    p = proba[:, 1] / (proba[:, 1] + proba[:, 2] + 0.0000001)

    if view == 'Y':
        t = track[(predict_y != 0)*(p > 0.01)]
        #t = track[(predict_y == 1)]
    else:
        predict_y = clf.predict_proba(data[all_cols].values)[:, 0]
        t = track[(predict_y < 0.9)]

    return score, lr, t




