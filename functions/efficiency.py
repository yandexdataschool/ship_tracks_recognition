import numpy as np
import pandas as pd


def one_event_efficiency(events, reco_events):
    """
    Get efficiencies for one event.
    :param pandas.DataFrame events: hits of the event.
    :param dict reco_events: dictionary of the reconstructed tracks before the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :return: lists of the tracks efficiencies for the Y, stereo views and whole station.
    """

    [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz] = reco_events

    efficiencies_y = []
    efficiencies_uv = []
    efficiencies = []

    # Efficiency for the Y view.
    for track in linking_table_yz:

        true_track_ids = events.loc[linking_table_yz[track]].TrackID.values

        _, unique_ids_counts = np.unique(true_track_ids, return_counts=True)

        eff = 1. * unique_ids_counts.max() / len(true_track_ids)
        efficiencies_y.append(eff)

    # Efficiency for the stereo views.
    efficiencies_uv = []
    for track in linking_table_xz:

        true_track_ids = events.loc[linking_table_xz[track]].TrackID.values

        _, unique_ids_counts = np.unique(true_track_ids, return_counts=True)

        eff = 1. * unique_ids_counts.max() / len(true_track_ids)
        efficiencies_uv.append(eff)

    # Efficiency for the station.
    for track in linking_table_xz:

        track_hits = list(linking_table_xz[track]) + list(linking_table_yz[track // 10000])
        true_track_ids = events.loc[track_hits].TrackID.values

        _, unique_ids_counts = np.unique(true_track_ids, return_counts=True)

        eff = 1. * unique_ids_counts.max() / len(true_track_ids)
        efficiencies.append(eff)

    return efficiencies_y, efficiencies_uv, efficiencies


def efficiency_per_track(event_ids, all_hits, reco_events):
    """
    Get efficiencies.
    :param list event_ids: list of event numbers.
    :param  pandas.DataFrame all_hits: strawtube hits.
    :param dict reco_events: dictionary of the reconstructed tracks before the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :return: lists of the tracks efficiencies for the Y, stereo views and whole station.
    """

    all_eff_y = []
    all_eff_stereo = []
    all_eff_station = []

    for event_id in event_ids:

        event = all_hits[all_hits['event'] == event_id]

        eff_y, eff_stereo, eff_station = one_event_efficiency(event, reco_events[event_id])

        all_eff_y += eff_y
        all_eff_stereo += eff_stereo
        all_eff_station += eff_station

    return all_eff_y, all_eff_stereo, all_eff_station


def view_check_reco_event(tracks_hits, pdg_codes_sets, all_hits):

    all_pdgs = []

    for track_id in tracks_hits.keys():

        pdgs, counts = np.unique(all_hits.loc[tracks_hits[track_id]][['PdgCode']].values, return_counts=True)
        track_pdg = pdgs#[counts == counts.max()][0]

        #all_pdgs.append(track_pdg)
        all_pdgs += list(track_pdg)

    #print all_pdgs
    check = 0
    for pdg_set in pdg_codes_sets:

        if len(set(np.unique(all_pdgs)) & set(pdg_set)) == len(pdg_set):
            check = 1

    return check


def efficiency_per_event(reconstructible_events, reco_events12, reco_events34, match_tracks, true_pdg_dict, all_hits):
    """
    Efficiencies per event after each stage of the tracks pattern recognition.
    :param reconstructible_events: dictionary of the reconstructed events.
                               Key - event number,
                               value - list of tracks ids.
    :param reco_events12: dictionary of the reconstructed tracks before the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :param reco_events34: dictionary of the reconstructed tracks after the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :param match_tracks: dictionary of the combined tracks before and after the magnet.
                               Key - event number,
                               value - [[track_id_12_stations, track_id_34_stations], track_id_12_stations, track_id_34_stations], ...].
    :param true_pdg_dict: dictionary of pdg codes of the tracks, where key - event number,
                            value - list of the true pdg codes before and after the magnet;
    :return: list of efficiencies per event after each stage.
    """

    n_reco_events = len(reconstructible_events.keys())

    n_events_y_12 = 0
    n_events_stereo_12 = 0
    n_events_station_12 = 0

    n_events_y_34 = 0
    n_events_stereo_34 = 0
    n_events_station_34 = 0

    n_events_combined_12_34 = 0
    n_events_matched = 0


    passed_event_ids = []
    missed_events = []

    # y_12
    for event_id in reconstructible_events.keys():

        tracks_y = reco_events12[event_id][0]
        check = view_check_reco_event(reco_events12[event_id][1], [[211, 13], [-211, -13]], all_hits)

        if len(tracks_y) > 1 and check == 1:
            n_events_y_12 += 1.
            passed_event_ids.append(event_id)

        else:
            missed_events.append(event_id)

    # stereo_12
    reco_events = passed_event_ids
    passed_event_ids = []

    for event_id in reco_events:

        tracks_stsreo = reco_events12[event_id][2]
        check = view_check_reco_event(reco_events12[event_id][3], [[211, 13], [-211, -13]], all_hits)

        if len(tracks_stsreo) > 1 and check == 1:
            n_events_stereo_12 += 1.
            passed_event_ids.append(event_id)

        else:
            missed_events.append(event_id)

    # station_12
    n_events_station_12 = n_events_stereo_12

    # y_34
    reco_events = passed_event_ids
    passed_event_ids = []

    for event_id in reco_events:

        tracks_stsreo = reco_events34[event_id][0]
        check = view_check_reco_event(reco_events34[event_id][1], [[211, 13], [-211, -13]], all_hits)

        if len(tracks_stsreo) > 1 and check == 1:
            n_events_y_34 += 1.
            passed_event_ids.append(event_id)

        else:
            missed_events.append(event_id)

    # stereo_34
    reco_events = passed_event_ids
    passed_event_ids = []

    for event_id in reco_events:

        tracks_stsreo = reco_events34[event_id][2]
        check = view_check_reco_event(reco_events34[event_id][3], [[211, 13], [-211, -13]], all_hits)

        if len(tracks_stsreo) > 1 and check == 1:
            n_events_stereo_34 += 1.
            passed_event_ids.append(event_id)

        else:
            missed_events.append(event_id)

    # station_34
    n_events_station_34 = n_events_stereo_34


    # combine 1&2/3&4
    reco_events = passed_event_ids
    passed_event_ids = []

    for event_id in reco_events:

        track_comb = match_tracks[event_id]

        if len(track_comb) > 1:
            n_events_combined_12_34 += 1
            passed_event_ids.append(event_id)

        else:
            missed_events.append(event_id)


    # matched
    reco_events = passed_event_ids
    passed_event_ids = []

    for event_id in reco_events:

        pdg_event = true_pdg_dict[event_id]

        n_mismatched = 0
        for pdg_track in pdg_event:

            if pdg_track[0] != pdg_track[1]:

                n_mismatched += 1

        if len(pdg_event) != 2:

            n_mismatched += 1

        if len(pdg_event) == 2 and pdg_event[0][0] == pdg_event[1][0]:

            n_mismatched += 1


        if n_mismatched == 0:
            n_events_matched += 1
            passed_event_ids.append(event_id)

        else:
            missed_events.append(event_id)



    n_events = [n_reco_events,
            n_events_y_12,
            n_events_stereo_12,
            n_events_station_12,
            n_events_y_34,
            n_events_stereo_34,
            n_events_station_34,
            n_events_combined_12_34,
            n_events_matched]

    n_events = np.array(n_events)

    return n_events, missed_events