import numpy
import pandas
from sklearn.linear_model import LinearRegression
import itertools

class WeightedRegression(object):
    
    def __init__(self, view='Y'):

        self.lr = None
        self.score = None
        self.score_std = None
        self.view = view

    def fit(self, hits):
        
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

        lr = LinearRegression()
        weights = (numpy.sqrt((1./Rx)**2 + (1./Ry)**2)).reshape(len(X), )
        lr.fit(X, Y, weights)
        self.lr = lr
        
        return lr

    def predict(self, X):
        
        return self.lr.predict(X)
    
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

        k = self.lr.coef_[0]

        Sign = (1. * (Y > self.lr.predict(X)) - 0.5) * 2.
        X_new = X + Sign * Rx * k/numpy.sqrt(1 + k**2)
        Y_new = Y - Sign * Ry * 1./numpy.sqrt(1 + k**2)

        dists = numpy.abs(self.lr.predict(X_new) - Y_new)/numpy.sqrt(k**2 + 1.0)

        return dists
    
def turbo_fit(track, event, clf, view):

    # y views
    lr = WeightedRegression(view=view)
    lr.fit(track)

    dists = lr.dist2track(track)[:, 0]

    data = pandas.DataFrame()
    data['min_dist'] = numpy.min(dists)
    data['max_dist'] = numpy.max(dists)
    data['score_std'] = numpy.std(dists)
    data['score'] = numpy.mean(dists)
    data['slope'] = lr.coef_[0][0]
    data['intercept'] = lr.intercept_[0]
    data['len'] = len(dists)

    predict_y = clf.predict(data.values)

    proba = clf.predict_proba(data.values)

    p = proba[0, 1] / (proba[0, 1] + proba[0, 2] + 0.0000001)

    if view == 'Y':
        t = track[(predict_y != 0)*(p > 0.01)]
        #t = track[(predict_y == 1)]
    else:
        predict_y = clf.predict_proba(data[all_cols].values)[:, 0]
        t = track[(predict_y < 0.9)]

    return score, lr, t