__author__ = 'mikhail91'

import numpy as np
import pandas as pd


def get_matched_tracks(reco_events12, reco_events34):
    """
    Match tracks reconstructed before and after the magnet.
    :param dict reco_events12: dictionary of the reconstructed tracks before the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :param dict reco_events34: dictionary of the reconstructed tracks after the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :return: dictionary of the matched tracks, where key - event number, value - list of the matched tracks ids;
             list of distances between the matched tracks on y;
             list of distances between the matched tracks on x.
    """

    dist_y = []
    dist_x = []
    maching_dict = {}

    for event_id in reco_events12.keys():

        tracks_yz_12 = reco_events12[event_id][0]
        tracks_xz_12 = reco_events12[event_id][2]

        tracks_yz_34 = reco_events34[event_id][0]
        tracks_xz_34 = reco_events34[event_id][2]

        maching_dict[event_id] = []

        for track_id_12 in tracks_xz_12.keys():

            [k_xz_12, b_xz_12] = tracks_xz_12[track_id_12]
            [k_yz_12, b_yz_12] = tracks_yz_12[track_id_12 // 10000]

            for track_id_34 in tracks_xz_34.keys():

                [k_xz_34, b_xz_34] = tracks_xz_34[track_id_34]
                [k_yz_34, b_yz_34] = tracks_yz_34[track_id_34 // 10000]

                y_12 = k_yz_12 * 3018. + b_yz_12
                y_34 = k_yz_34 * 3018. + b_yz_34

                x_12 = k_xz_12 * 3018. + b_xz_12
                x_34 = k_xz_34 * 3018. + b_xz_34

                if (np.abs(y_12 - y_34) <= 2.) and (np.abs(x_12 - x_34) <= 20.):

                    maching_dict[event_id].append([track_id_12, track_id_34])
                    dist_y.append(y_12 - y_34)
                    dist_x.append(x_12 - x_34)

    return maching_dict, dist_y, dist_x