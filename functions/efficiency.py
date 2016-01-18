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

        track_hits = linking_table_xz[track] + linking_table_yz[track // 10000]
        true_track_ids = events.loc[track_hits].TrackID.values

        _, unique_ids_counts = np.unique(true_track_ids, return_counts=True)

        eff = 1. * unique_ids_counts.max() / len(true_track_ids)
        efficiencies.append(eff)

    return efficiencies_y, efficiencies_uv, efficiencies


def efficiency(event_ids, all_hits, reco_events):
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