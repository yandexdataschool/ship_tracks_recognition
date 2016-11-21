__author__ = 'mikhail91'


from copy import deepcopy
import numpy

class RANSACTracker(object):

    def __init__(self, n_tracks, min_hits, regressor):
        """
        This class uses RANSAC linear regression for the tracks reconstruction.
        :param n_tracks: int, number of tracks are searching for.
        :param min_hits: int, min number of hits in a track.
        :param regressor: scikit-learn RANSACRegressor class object.
        :return:
        """

        self.n_tracks = n_tracks
        self.min_hits = min_hits
        self.regressor = regressor

        self.labels_ = None
        self.tracks_params_ = None

    def fit(self, x, y, sample_weight=None):
        """
        Search for the tracks.
        :param x: list of floats, x-coordinates of the hits.
        :param y: ist of floats, y-coordinates of the hits.
        :param sample_weight: sample_weight: list of floats, weights of the hits.
        :return:
        """


        labels = -1 * numpy.ones(len(x))
        tracks_params = []
        indexes = numpy.arange(len(x))

        for track_id in range(self.n_tracks):

            x_track = x[labels == -1]
            y_track = y[labels == -1]
            indexes_track = indexes[labels == -1]

            if len(x_track) < self.min_hits or len(x_track) <= 0:
                break

            flag = 0
            while flag != 1 and flag > -100:

                try:

                    regressor = deepcopy(self.regressor)
                    regressor.fit(x_track.reshape(-1, 1), y_track)
                    inlier_mask = regressor.inlier_mask_
                    estimator = regressor.estimator_


                    if (inlier_mask * 1).sum() >= self.min_hits:

                        labels[indexes_track[inlier_mask]] = track_id
                        tracks_params.append([estimator.coef_[0], estimator.intercept_])
                    flag = 1

                except:

                    flag += -1




        self.tracks_params_ = numpy.array(tracks_params)
        self.labels_ = labels
