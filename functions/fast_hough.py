__author__ = 'mikhail91'

from sklearn.linear_model import LinearRegression
import numpy

class FastHough(object):

    def __init__(self, n_tracks=None, min_hits=4, k_size=0.1, b_size=10, k_limits=(-0.3, 0.3), b_limits=(-800, 800)):


        self.n_tracks = n_tracks
        self.min_hits = min_hits

        self.k_size = k_size
        self.b_size = b_size

        self.k_limits = k_limits
        self.b_limits = b_limits

    def transform(self, x, y):

        track_inds = []

        for first in range(0, len(x)):
            for second in range(first, len(x)):

                x1, y1 = x[first], y[first]
                x2, y2 = x[second], y[second]

                if x1 == x2:
                    continue

                k = 1. * (y2 - y1) / (x2 - x1)
                b = y1 - k * x1

                if k >= self.k_limits[0] and k <= self.k_limits[1] and b >= self.b_limits[0] and b <= self.b_limits[1]:

                    one_track_inds = self.hits_in_bin(x, y, k, b)

                    if len(one_track_inds) >= self.min_hits:
                        track_inds.append(one_track_inds)

        return numpy.array(track_inds)


    def hits_in_bin(self, x, y, k_bin, b_bin):


        b_left = y - (k_bin - 0.5 * self.k_size) * x
        b_right = y - (k_bin + 0.5 * self.k_size) * x

        inds = numpy.arange(0, len(x))

        sel = (b_left >= b_bin - 0.5 * self.b_size) * (b_right <= b_bin + 0.5 * self.b_size) + \
              (b_left <= b_bin + 0.5 * self.b_size) * (b_right >= b_bin - 0.5 * self.b_size)

        track_inds = inds[sel]

        return track_inds



    def get_hit_labels(self, track_inds, n_hits):

        labels = -1. * numpy.ones(n_hits)
        used = numpy.zeros(n_hits)
        track_id = 0
        n_tracks = 0


        while 1:

            track_lens = numpy.array([len(i[used[i] == 0]) for i in track_inds])

            if len(track_lens) == 0:
                break

            max_len = track_lens.max()

            if max_len < self.min_hits:
                break

            one_track_inds = track_inds[track_lens == track_lens.max()][0]
            one_track_inds = one_track_inds[used[one_track_inds] == 0]

            used[one_track_inds] = 1
            labels[one_track_inds] = track_id
            track_id += 1

            n_tracks += 1
            if self.n_tracks != None and n_tracks >= self.n_tracks:
                break

        return numpy.array(labels)

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

            lr = LinearRegression(copy_X=False)
            lr.fit(x_track.reshape(-1,1), y_track, sample_weight_track)

            params = list(lr.coef_) + [lr.intercept_]
            tracks_params.append(params)

        return numpy.array(tracks_params)


    def fit(self, x, y, sample_weight=None):

        track_inds = self.transform(x, y)

        self.track_inds_ = track_inds
        self.labels_ = self.get_hit_labels(self.track_inds_, len(x))
        self.tracks_params_ = self.get_tracks_params(x, y, self.labels_, sample_weight)

