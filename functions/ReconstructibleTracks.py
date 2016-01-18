__author__ = 'mikhail91'

import numpy as np
import pandas as pd

def get_reconstractible_tracks(event_id, all_hits, all_mctracks, all_velo_points):
    """
    Get all reconstructible tracks of an event.
    :param int event_id: number of the event.
    :param pandas.DataFrame all_hits: strawtube hits.
    :param pandas.DataFrame all_mctracks: monte carlo tracks.
    :param pandas.DataFrame all_velo_points: hits in Veto station.
    :return: list of reconstructible tracks.
    """

    mctrack_ids = []

    hits = all_hits[all_hits.event == event_id]
    mctracks = all_mctracks[all_mctracks.event == event_id]
    velo_points = all_velo_points[all_velo_points.event == event_id]

    # 1. Event must has less than 500 hits.
    if len(hits) > 500:
        return []

    # 2. mctrack_ids: list of tracks decaying after the last station and originating before the first one
    for track_id in range(0, len(mctracks)):

        if mctracks.iloc[[track_id]].StartZ.values[0] > 3557:

            mather_track_id = mctracks.iloc[[track_id]].MotherId.values[0]

            if mather_track_id > -1:

                mother_track_z = mctracks.iloc[[mather_track_id]].StartZ.values[0]

                if mother_track_z > -1978 and mother_track_z < 2580:

                    if mather_track_id not in mctrack_ids:

                        mctrack_ids.append(mather_track_id)

    # 3. veto_track_ids: list of tracks with hits in TimeDet
    veto_track_ids = []

    for hit_id in range(0, len(velo_points)):

        hit_z = velo_points.iloc[[hit_id]].Z.values[0]

        if hit_z == 3688:

            track_id = velo_points.iloc[[hit_id]].TrackID.values[0]

            if track_id not in veto_track_ids:

                veto_track_ids.append(track_id)

    # 4. Remove tracks from mctrack_ids that are not in veto_track_ids
    for track_id in mctrack_ids:

        if track_id not in veto_track_ids:

            if track_id in mctrack_ids:
                mctrack_ids.remove(track_id)

    # 5. Find straws that have multiple hits. Remove duplicated hits.
    duplicated_hit_inds = []

    hits['DetectorID'] = hits.StrawNb.values + \
                         2000 + \
                         10000 * hits.LayerNb.values + \
                         100000 * hits.PlaneNb.values + \
                         1000000 * hits.ViewNb.values + \
                         10000000 * hits.StatNb.values

    hits_stats = hits[hits.StatNb != 5]

    for hit_ind in hits_stats.index:

        detector_id = hits_stats.loc[[hit_ind]].DetectorID.values[0]
        selected_hits = hits_stats[hits_stats.DetectorID == detector_id]
        duplicated_ind = selected_hits[selected_hits.X != selected_hits.X.min()].index

        duplicated_hit_inds += list(duplicated_ind)


    new_index = list(hits_stats.index)

    for index in duplicated_hit_inds:

        if index in new_index:

            if index in new_index:
                new_index.remove(index)

    new_hits = hits_stats.loc[new_index]

    # 6. Remove tracks that are not in the acceptance ellipse.
    for track_id in mctrack_ids:

        track_hits = new_hits[new_hits.TrackID == track_id]

        for stat_nb in [1, 2, 3, 4]:

            stat_hits = track_hits[track_hits.StatNb == stat_nb]

            if len(stat_hits) == 0 or ((stat_hits.X)**2 + (stat_hits.Y)**2 >= 1.).sum() == 0:

                if track_id in mctrack_ids:
                    mctrack_ids.remove(track_id)

    # 7. Find pion and muon tracks.
    pion = 0
    muon = 0

    for track_id in mctrack_ids:

        if np.abs(mctracks.iloc[[track_id]].PdgCode.values[0]) == 13:
            muon = 1

        if np.abs(mctracks.iloc[[track_id]].PdgCode.values[0]) == 211:
            pion = 1

    if pion != 1 or muon != 1:
        mctrack_ids = []

    return mctrack_ids
