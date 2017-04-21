__author__ = 'Mikhail Hushchyn'

import numpy


class HitsMatchingEfficiency(object):
    def __init__(self, eff_threshold=0.5, n_tracks=None):
        """
        This class calculates tracks efficiencies, reconstruction efficiency, ghost rate and clone rate for one event using hits matching.
        :param eff_threshold: float, threshold value of a track efficiency to consider a track reconstructed.
        :return:
        """

        self.eff_threshold = eff_threshold
        self.n_tracks = n_tracks

    def fit(self, true_labels, track_inds):
        """
        The method calculates all metrics.
        :param true_labels: numpy.array, true labels of the hits.
        :param track_inds: numpy.array, hits of recognized tracks.
        :return:
        """

        # Calculate efficiencies
        efficiencies = []
        tracks_id = []

        for one_track_inds in track_inds:

            track = true_labels[one_track_inds]
            # if len(track[track != -1]) == 0:
            #    continue
            unique, counts = numpy.unique(track, return_counts=True)

            if len(track) != 0:
                eff = 1. * counts.max() / len(track)
                efficiencies.append(eff)
                tracks_id.append(unique[counts == counts.max()][0])

        tracks_id = numpy.array(tracks_id)
        efficiencies = numpy.array(efficiencies)
        self.efficiencies_ = efficiencies

        # Calculate avg. efficiency
        avg_efficiency = efficiencies.mean()
        self.avg_efficiency_ = avg_efficiency

        # Calculate reconstruction efficiency
        true_tracks_id = numpy.unique(true_labels)

        if self.n_tracks == None:
            n_tracks = (true_tracks_id != -1).sum()
        else:
            n_tracks = self.n_tracks

        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        if n_tracks != 0:
            recognition_efficiency = 1. * len(unique) / (n_tracks)
        else:
            recognition_efficiency = 0
        self.recognition_efficiency_ = recognition_efficiency

        # Calculate ghost rate
        if n_tracks != 0:
            ghost_rate = 1. * (len(tracks_id) - len(reco_tracks_id[reco_tracks_id != -1])) / (n_tracks)
        else:
            ghost_rate = 0
        self.ghost_rate_ = ghost_rate

        # Calculate clone rate
        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        if n_tracks != 0:
            clone_rate = (counts - numpy.ones(len(counts))).sum() / (n_tracks)
        else:
            clone_rate = 0
        self.clone_rate_ = clone_rate