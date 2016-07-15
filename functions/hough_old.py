__author__ = 'mikhail91'

import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class LinearHoughModel(object):

    def __init__(self, k_params, b_params, n_candidates, weights_mul=10., show=False):
        """
        This calss is realizarion of the Linear Hough Transform method for the track recognition.
        :param k_params: tuple (min, max, n_bins), bins parameters for the k parameter.
        :param b_params: tuple (min, max, n_bins), bins parameters for the b parameter.
        :param n_candidates: int, number of tracks searching for.
        :param weights_mul: float, weights of a track's bins will be decreased by this values.
        :param show: boolean, if True show hough transfrom pictures for the each track.
        :return:
        """

        self.k_params = k_params
        self.b_params = b_params
        self.n_candidates = n_candidates
        self.weights_mul = weights_mul
        self.show = show

    def _hough_transform(self, x, y, k_params, b_params):
        """
        The Hough transfrom for the one hit.
        :param x: float, x-coordinate of a hit.
        :param y: float, y-coordinate of a hit.
        :param k_params: tuple (min, max, n_bins), bins parameters for the k parameter.
        :param b_params: tuple (min, max, n_bins), bins parameters for the b parameter.
        :return: numpy.array, shape=(n_points,); numpy.array, shape=(n_points,)
        """

        X_hough = numpy.linspace(k_params[0], k_params[1], 10 * k_params[2])
        Y_hough = -x * X_hough + y

        return X_hough, Y_hough

    def _fit_one(self, X, Y, labels, num_candidate, weights_mul=10.):
        """
        Searching for one track.
        :param X: numpy.array, shape=(n_hits, ), x-coordinates of the hits.
        :param Y: numpy.array, shape=(n_hits, ), y-coordinates of the hits.
        :param labels: numpy.array, shape=(n_hits, ), labels of the hits. -1 means unclassifier hit.
        0, 1, 2, ... mean thrack ids.
        :param num_candidate: int, label of a new track.
        :param weights_mul: float, weights of a new track's bins will be decreased by this values.
        :return: numpy.array, shape=(n_hits, )
        """

        X_hough_all = []
        Y_hough_all = []
        weights_all = []
        ind_all= []

        for ind, (x, y, lab) in enumerate(zip(X, Y, labels)):

            if lab == 0:

                X_hough, Y_hough = self._hough_transform(x, y, self.k_params, self.b_params)
                X_hough_all += list(X_hough.reshape(-1))
                Y_hough_all += list(Y_hough.reshape(-1))
                weights_all += [1.] * len(Y_hough.reshape(-1))
                ind_all += [ind]*len(X_hough.reshape(-1))

            elif lab != 0 and weights_mul != None:

                X_hough, Y_hough = self._hough_transform(x, y, self.k_params, self.b_params)
                X_hough_all += list(X_hough.reshape(-1))
                Y_hough_all += list(Y_hough.reshape(-1))
                weights_all += [1./weights_mul] * len(Y_hough.reshape(-1))
                ind_all += [ind]*len(X_hough.reshape(-1))


        XY_hough = numpy.concatenate((numpy.array(X_hough_all).reshape((-1,1)),
                                      numpy.array(Y_hough_all).reshape((-1,1)),
                                      numpy.array(ind_all).reshape((-1,1))), axis=1)
        weights = numpy.array(weights_all).reshape(-1)



        if self.show==True:
            plt.figure(figsize=(10, 7))
            (counts, xedges, yedges, _) = plt.hist2d(x=XY_hough[:,0], y=XY_hough[:,1], weights=weights,
                                                     range=[[self.k_params[0], self.k_params[1]],
                                                            [self.b_params[0], self.b_params[1]]],
                                                    bins=[self.k_params[2], self.b_params[2]])
            plt.colorbar()

        elif self.show==False:
            (counts, xedges, yedges) = numpy.histogram2d(x=XY_hough[:,0], y=XY_hough[:,1], weights=weights,
                                                         range=[[self.k_params[0], self.k_params[1]],
                                                            [self.b_params[0], self.b_params[1]]],
                                                         bins=[self.k_params[2], self.b_params[2]])

        if self.show:
            plt.show()
        else:
            pass
            # plt.clf()
            # plt.close()


        k_max_ind, b_max_ind = numpy.unravel_index(indices=counts.argmax(), dims=counts.shape)

        k_min, k_max = xedges[k_max_ind:k_max_ind+2]
        b_min, b_max = yedges[b_max_ind:b_max_ind+2]

        sel = (XY_hough[:,0] >= k_min) * (XY_hough[:,0] < k_max) * \
              (XY_hough[:,1] >= b_min) * (XY_hough[:,1] < b_max)
        XY_hough_cand = XY_hough[sel]

        ind_cand = list(numpy.unique(XY_hough_cand[:, 2]))
        ind_cand = numpy.array(ind_cand)

        labels[ind_cand.astype(int)] = num_candidate

        return labels

    def get_tracks_params(self, x, y, labels, sample_weight=None):

        tracks_params = []

        unique_labels = numpy.unique(labels)
        track_ids = unique_labels[unique_labels != -1]

        if len(track_ids) == 0:
            return []

        for track_id in track_ids:

            x_track = x[labels == track_id]
            y_track = y[labels == track_id]

            if sample_weight != None:
                sample_weight_track = sample_weight[labels == track_id]
            else:
                sample_weight_track = None

            lr = LinearRegression()
            lr.fit(x_track.reshape(-1,1), y_track, sample_weight_track)

            params = list(lr.coef_) + [lr.intercept_]
            tracks_params.append(params)

        return numpy.array(tracks_params)

    def fit(self, X, Y, sample_weight=None):
        """
        Searching for all tracks.
        :param X: numpy.array, shape=(n_hits, ), x-coordinates of the hits.
        :param Y: numpy.array, shape=(n_hits, ), y-coordinates of the hits.
        :param sample_weight: numpy.array, shape=(n_hits, ), weights of each hit for linear model fit.
        """

        labels = numpy.zeros(len(X))

        for num in range(1, self.n_candidates + 1):

            labels = self._fit_one(X, Y, labels, num, self.weights_mul)


        self.labels_ = numpy.array(labels) - 1.
        self.tracks_params_ = self.get_tracks_params(X, Y, self.labels_, sample_weight)

