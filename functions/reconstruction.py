__author__ = 'mikhail91'

from copy import copy
import numpy
import pandas

class TracksReconstruction2D(object):

    def __init__(self, model_y, model_stereo, unique_hit_labels=True):
        """
        This is realization of the reconstruction scheme which uses two 2D projections to reconstruct a 3D track.
        :param model_y: model for the tracks reconstruction in y-z plane.
        :param model_stereo: model for the tracks reconstruction in x-z plane.
        :return:
        """

        self.model_y = copy(model_y)
        self.model_stereo = copy(model_stereo)
        self.unique_hit_labels = unique_hit_labels

        self.labels_ = None
        self.tracks_params_ = None

    def get_xz(self, plane_k, plane_b, event):
        """
        This method returns (z, x) coordinated of the intersections of the straw tubes in stereo-views and
        a plane corresponding to a founded track in y-view.
        :param plane_k: float, slope of the track in y-view.
        :param plane_b: float, intercept of the track in y-view.
        :param event: pandas.DataFrame, event which contains information about active straw tubes.
        :return: z, x coordinates of the intersections.
        """

        Wz1 = event.Wz1.values
        Wx1 = event.Wx1.values
        Wx2 = event.Wx2.values
        Wy1 = event.Wy1.values
        Wy2 = event.Wy2.values

        y = plane_k * Wz1 + plane_b
        x = (Wx2 - Wx1) / (Wy2 - Wy1) * (y - Wy1) + Wx1

        return Wz1, x

    def fit(self, event, sample_weight=None):
        """
        Fit of the models.
        :param event: pandas.DataFrame, event which contains information about active straw tubes.
        :param sample_weight: numpy.array shape=[n_hits], weight of each hits.
        :return:
        """

        self.track_inds_ = []
        self.tracks_params_ = []

        indeces = numpy.arange(len(event))

        # Tracks Reconstruction in Y-view
        event_y = event[event.IsStereo.values == 0]
        indeces_y = indeces[event.IsStereo.values == 0]
        mask_y = event.IsStereo.values == 0

        x_y = event_y.Wz1.values
        y_y = event_y.Wy1.values

        if sample_weight != None:
            sample_weight_y = sample_weight[mask_y == 1]
        else:
            sample_weight_y = None


        self.model_y.fit(x_y, y_y, sample_weight_y)
        track_inds_y = self.model_y.track_inds_
        for i in range(len(track_inds_y)):
            track_inds_y[i] = indeces_y[track_inds_y[i]]
        tracks_params_y = self.model_y.tracks_params_

        # Tracks Reconstruction in Stereo_views
        event_stereo = event[event.IsStereo.values == 1]
        indeces_stereo = indeces[event.IsStereo.values == 1]
        used = numpy.zeros(len(event_stereo))
        mask_stereo = event.IsStereo.values == 1

        for track_id, one_track_y in enumerate(tracks_params_y):

            if len(one_track_y) != 0:

                plane_k, plane_b = one_track_y
                x_stereo, y_stereo = self.get_xz(plane_k, plane_b, event_stereo)

                if self.unique_hit_labels:
                    sel = (used==0) * (numpy.abs(y_stereo) <= 293.)
                else:
                    sel = (numpy.abs(y_stereo) <= 293.)

                if sample_weight != None:
                    sample_weight_stereo = sample_weight[mask_stereo == 1][sel]
                else:
                    sample_weight_stereo = None

                self.model_stereo.fit(x_stereo[sel], y_stereo[sel], sample_weight_stereo)
                if len(self.model_stereo.track_inds_) == 0:
                    continue
                track_inds_stereo = self.model_stereo.track_inds_
                for i in range(len(track_inds_stereo)):
                    inds = numpy.arange(len(used))[sel][track_inds_stereo[i]]
                    used[inds] = 1
                    track_inds_stereo[i] = indeces_stereo[sel][track_inds_stereo[i]]
                tracks_params_stereo = self.model_stereo.tracks_params_


            else:

                one_track_stereo = []


            for i in range(len(tracks_params_stereo)):
                self.tracks_params_.append([one_track_y, tracks_params_stereo[i]])
                self.track_inds_.append([track_inds_y[track_id], track_inds_stereo[i]])

        self.tracks_params_ = numpy.array(self.tracks_params_)
        self.track_inds_ = numpy.array(self.track_inds_)
