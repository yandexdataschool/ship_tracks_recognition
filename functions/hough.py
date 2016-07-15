__author__ = 'mikhail91'

import numpy
from sklearn.linear_model import LinearRegression


class LinearHoughModel(object):

    def __init__(self, k_params=(-2, 2, 0.1), b_params=(-10, 10, 1), min_hits=4, multiplier=10.):
        """
        This calss is realizarion of the Linear Hough Transform method for the track recognition.
        :param k_params: tuple (min, max, step), bins parameters for the k parameter.
        :param b_params: tuple (min, max, step), bins parameters for the b parameter.
        :param min_hits: int, tracks with number of hits larger then min_hits considered as a track candidate.
        :return:
        """

        self.k_params = k_params
        self.b_params = b_params
        self.min_hits = min_hits
        self.multiplier = multiplier

        #self.pool = Pool(3)

        self.labels_ = None

    def linear_hough(self, x_hit, y_hit, k_params, b_params):
        """
        This method do Hough Transform just for one point.
        :param x_hit: float, x-coordinate of a hit. y = kx + b.
        :param y_hit: float, y-coordinate of a hit. y = kx + b.
        :param k_params: tuple (min, max, step), bins parameters for the k parameter.
        :param b_params: tuple (min, max, step), bins parameters for the b parameter.
        :return: numpy.array; numpy.array.
        """

        # y = kx+b -> b = y - kx

        # Change the following code by the correct one. Inputs and outputs are the same.
        # k = numpy.random.rand(100)
        # b = numpy.random.rand(100)

        n_points = self.multiplier * (k_params[1] - k_params[0]) / (k_params[2])

        k = numpy.arange(*k_params)
        b = y_hit - k * x_hit

        return k, b

    def get_hits(self, hists, max_ind):
        """
        This method finds hits that corresponding cell of histogram with max counts.
        :param hists: list of histograms for each hit.
        :param max_ind: tuple (int, int), index of the cell with max counts.
        :return: list of indeces of the hits.
        """

        # Change the following code by the correct one. Inputs and outputs are the same.
        # hits = numpy.random.randint(0, len(hists), 3)

        hits = []

        for ind, hit_hist in enumerate(hists):

            if hit_hist[max_ind] == 1:

                hits.append(ind)

        return hits

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


    def fit(self, x, y, sample_weight=None):
        """
        This method runs the Linear Hough Transfrom method.
        :param x: numpy.array, shape = [n_hits], x-coordinates of the hits.
        :param y: numpy.array, shape = [n_hits], y-coordinates of the hits.
        :return:
        """

        # Finish the function. Inputs and outputs are the same.

        n_hits = len(x)
        labels = -1. * numpy.ones(n_hits)

        track_id = 0

        hists = []
        n_k = (self.k_params[1] - self.k_params[0]) / (self.k_params[2])
        n_b = (self.b_params[1] - self.b_params[0]) / (self.b_params[2])

        for hit_id in range(n_hits):

            x_hit = x[hit_id]
            y_hit = y[hit_id]

            k_hit, b_hit = self.linear_hough(x_hit, y_hit, self.k_params, self.b_params)

            (hit_hist, xedges, yedges) = numpy.histogram2d(k_hit, b_hit, range=[[self.k_params[0], self.k_params[1]],
                                                            [self.b_params[0], self.b_params[1]]],
                                                         bins=[n_k, n_b])
            hit_hist = (hit_hist > 0) * 1.
            hists.append(hit_hist)


        hists = numpy.array(hists)
        self.hists = hists
        total_hist = hists.sum(axis=0)


        while total_hist.max() >= self.min_hits:

            total_hist = hists[labels == -1].sum(axis=0)
            max_ind = numpy.unravel_index(numpy.argmax(total_hist), total_hist.shape)

            hits = self.get_hits(hists, max_ind)

            if len(hits) >= self.min_hits:

                labels[hits] = track_id
                track_id += 1


        self.labels_ = labels
        self.tracks_params_ = self.get_tracks_params(x, y, labels, sample_weight)
