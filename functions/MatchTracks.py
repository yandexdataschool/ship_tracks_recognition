__author__ = 'mikhail91'

import numpy as np
import pandas as pd


def get_dx_dy(track12, track34, zmagnet):
    """
    Get dx and dy values between the tracks of the 1&2 and 3&4 stations in center of the magnet.
    :param track12: list of tack parameters for xz and yz planes in the 1&2 stations.
    :param track34: list of tack parameters for xz and yz planes in the 3&4 stations.
    :param zmagnet: z coordinate of the center of the magnet
    :return: dx, dy values
    """

    [k_xz_12, b_xz_12] = track12[0]
    [k_yz_12, b_yz_12] = track12[1]

    [k_xz_34, b_xz_34] = track34[0]
    [k_yz_34, b_yz_34] = track34[1]

    y_12 = k_yz_12 * zmagnet + b_yz_12
    y_34 = k_yz_34 * zmagnet + b_yz_34

    x_12 = k_xz_12 * zmagnet + b_xz_12
    x_34 = k_xz_34 * zmagnet + b_xz_34

    dy = (y_12 - y_34)
    dx = (x_12 - x_34)

    return dx, dy

def get_charge(track12, track34):

    """
    Get charge of the track's particle.
    :param track12: list of tack parameters for xz and yz planes in the 1&2 stations.
    :param track34: list of tack parameters for xz and yz planes in the 3&4 stations.
    :return: charge value.
    """

    [k_yz_12, b_yz_12] = track12[1]
    [k_yz_34, b_yz_34] = track34[1]

    difftan = (k_yz_12 - k_yz_34) / (1. + k_yz_12 * k_yz_34)

    if difftan > 0:

        charge = -1.

    else:

        charge = 1.

    return charge

def get_pinv(track12, track34, Bm):
    """
    Get inversal momentum value of the track.
    :param track12: list of tack parameters for xz and yz planes in the 1&2 stations.
    :param track34: list of tack parameters for xz and yz planes in the 3&4 stations.
    :param Bm: iduction of the magnetic field in mT
    :return: inversal momentum value
    """

    [k_yz_12, b_yz_12] = track12[1]
    [k_yz_34, b_yz_34] = track34[1]

    a = np.arctan(k_yz_12)
    b = np.arctan(k_yz_34)
    pinv = np.sin(a - b) / (0.3 * Bm)

    return pinv

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
             dictionary of charge values of the tracks, where key - event number, value - list of charge values;
             dictionary of inverse momentum values of the tracks, where key - event number, value - list of inverse momentum values;
             list of distances between the matched tracks on y;
             list of distances between the matched tracks on x.
    """

    dist_y = []
    dist_x = []
    maching_dict = {}
    charge_dict = {}
    pinv_dict = {}

    Bm = -0.75
    zmagnet = 3070.

    for event_id in reco_events12.keys():

        tracks_yz_12 = reco_events12[event_id][0]
        tracks_xz_12 = reco_events12[event_id][2]

        tracks_yz_34 = reco_events34[event_id][0]
        tracks_xz_34 = reco_events34[event_id][2]

        maching_dict[event_id] = []
        charge_dict[event_id] = []
        pinv_dict[event_id] = []

        for track_id_12 in tracks_xz_12.keys():

            track12_x = tracks_xz_12[track_id_12]
            track12_y = tracks_yz_12[track_id_12 // 10000]
            track12 = [track12_x, track12_y]

            for track_id_34 in tracks_xz_34.keys():

                track34_x = tracks_xz_34[track_id_34]
                track34_y = tracks_yz_34[track_id_34 // 10000]
                track34 = [track34_x, track34_y]

                dx, dy = get_dx_dy(track12, track34, zmagnet)

                if (np.abs(dy) <= 2.) and (np.abs(dx) <= 20.):

                    # distance
                    maching_dict[event_id].append([track_id_12, track_id_34])
                    dist_y.append(dy)
                    dist_x.append(dx)

                    # charge
                    charge = get_charge(track12, track34)
                    charge_dict[event_id].append(charge)

                    # momentum inv
                    pinv = get_pinv(track12, track34, Bm)
                    pinv_dict[event_id].append(pinv)




    return maching_dict, charge_dict, pinv_dict, dist_y, dist_x



def get_true_charge(hits):
    """
    Get true charge values for tracks before and after the magnet.
    :param hits: pandas.DataFrame, hits of the track
    :return: true charges and pdg codes
    """

    pdg_uniques, pdg_counts = np.unique(hits.PdgCode.values, return_counts=True)

    pdg = pdg_uniques[pdg_counts == pdg_counts.max()][0]

    charge = 0.

    if pdg == 13 or pdg == -211:

        charge = -1.

    elif pdg == -13 or pdg == 211:

        charge = 1.

    return charge, pdg

def get_true_pinv(hits):
    """
    Get true inversal momentrum value for the tracks before and after the magnet.
    :param hits: pandas.DataFrame, hits of the track
    :return: true inversal momentrum  values
    """

    trackid_uniques, trackid_counts = np.unique(hits.TrackID.values, return_counts=True)

    track_id = trackid_uniques[trackid_counts == trackid_counts.max()][0]

    px = hits.Px.values[hits.TrackID == track_id]
    py = hits.Py.values[hits.TrackID == track_id]
    pz = hits.Pz.values[hits.TrackID == track_id]

    pinv = 1. / np.sqrt(px**2 + py**2 + pz**2)

    return pinv.mean()


def get_true_match(reco_events12, reco_events34, match_tracks, all_hits):
    """
    Get true charge values, pdg codes and inversal momentums for tracks before and after the magnet.
    :param dict reco_events12: dictionary of the reconstructed tracks before the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :param dict reco_events34: dictionary of the reconstructed tracks after the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :param match_tracks: dictionary of the combined tracks before and after the magnet.
                               Key - event number,
                               value - [[track_id_12_stations, track_id_34_stations], track_id_12_stations, track_id_34_stations], ...].
    :param all_hits: pandas.DataFrame, MC data for all hits.
    :return: dictionary of the charge values, where key - event number,
             value - list of the true charges before and after the magnet;
             dictionary of pdg codes of the tracks, where key - event number,
             value - list of the true pdg codes before and after the magnet;
             dictionary of inverse momentum values of the tracks,
             where key - event number, value - list of inverse momentum values before and after the magnet;
    """

    true_charge_dict = {}
    true_pdg_dict = {}
    true_pinv_dict = {}

    for event_id in match_tracks.keys():

        true_charge_dict[event_id] = []
        true_pdg_dict[event_id] = []
        true_pinv_dict[event_id] = []

        for track in match_tracks[event_id]:

            [track12_ids, track34_ids] = track

            [track12_yz_id, track12_xz_id] = [track12_ids // 10000, track12_ids]
            hits12_yz_ids = list(reco_events12[event_id][1][track12_yz_id])
            hits12_xz_ids = list(reco_events12[event_id][3][track12_xz_id])
            hits12_ids = hits12_yz_ids + hits12_xz_ids
            hits12 = all_hits.loc[hits12_ids]

            [track34_yz_id, track34_xz_id] = [track34_ids // 10000, track34_ids]
            hits34_yz_ids = list(reco_events34[event_id][1][track34_yz_id])
            hits34_xz_ids = list(reco_events34[event_id][3][track34_xz_id])
            hits34_ids = hits34_yz_ids + hits34_xz_ids
            hits34 = all_hits.loc[hits34_ids]

            hits = all_hits.loc[hits12_ids + hits34_ids]

            charge12, pdg12 = get_true_charge(hits12)
            charge34, pdg34 = get_true_charge(hits34)

            true_charge_dict[event_id].append([charge12, charge34])
            true_pdg_dict[event_id].append([pdg12, pdg34])

            true_pinv12 = get_true_pinv(hits12)
            true_pinv34 = get_true_pinv(hits34)
            true_pinv_dict[event_id].append([true_pinv12, true_pinv34])


    return true_charge_dict, true_pdg_dict, true_pinv_dict


def get_pinv_true_pinv(pinv_dict, true_pinv_dict, true_charge_dict):
    """
    Get arrays of inversal momentum and MC true inversal momentum of the tracks
    :param pinv_dict: dictionary of inverse momentum values of the tracks,
                      where key - event number, value - list of inverse momentum values;
    :param true_pinv_dict: dictionary of true inverse momentum values of the tracks,
                           where key - event number, value - list of inverse momentum values;
    :param true_charge_dict: dictionary of the true charge values, where key - event number,
                              value - list of the true charges before and after the magnet;
    :return: arrays of inversal momentum and MC true inversal momentum.
    """

    pinv = []
    for key in pinv_dict.keys():
        pinv_event = pinv_dict[key]
        pinv += pinv_event
    pinv = np.array(pinv)

    true_pinv = []
    for key in pinv_dict.keys():
        pinv_event = true_pinv_dict[key]
        charge_event = true_charge_dict[key]
        for pinv_track, charge_track in zip(pinv_event, charge_event):
            true_pinv += [pinv_track[1]*charge_track[1]]
    true_pinv = np.array(true_pinv)

    return pinv, true_pinv


def get_true_label(hits12, hits34, all_hits):

    track_ids_12 = all_hits.loc[hits12].TrackID.values
    track_ids_34 = all_hits.loc[hits34].TrackID.values

    track_ids_12_uniques, track_ids_12_counts = np.unique(track_ids_12, return_counts=True)
    true_track_id_12 = track_ids_12_uniques[track_ids_12_counts == track_ids_12_counts.max()][0]

    track_ids_34_uniques, track_ids_34_counts = np.unique(track_ids_34, return_counts=True)
    true_track_id_34 = track_ids_34_uniques[track_ids_34_counts == track_ids_34_counts.max()][0]

    if true_track_id_12 == true_track_id_34:
        true_label = 1
    else:
        true_label = 0

    return true_label


def get_matching_data(reco_events12, reco_events34, all_hits):
    """
    """

    data = pd.DataFrame(columns=['EventID',
                                     'dx', 'dy',
                                     'k_xz_12', 'b_xz_12',
                                     'k_yz_12', 'b_yz_12',
                                     'k_xz_34', 'b_xz_34',
                                     'k_yz_34', 'b_yz_34',
                                     'label',
                                     'track_id_12', 'track_id_34'])

    Bm = -0.75
    zmagnet = 3070.

    for event_id in reco_events12.keys():

        tracks_yz_12 = reco_events12[event_id][0]
        hits_yz_12 = reco_events12[event_id][1]
        tracks_xz_12 = reco_events12[event_id][2]
        hits_xz_12 = reco_events12[event_id][3]

        tracks_yz_34 = reco_events34[event_id][0]
        hits_yz_34 = reco_events34[event_id][1]
        tracks_xz_34 = reco_events34[event_id][2]
        hits_xz_34 = reco_events34[event_id][3]



        for track_id_12 in tracks_xz_12.keys():

            track12_x = tracks_xz_12[track_id_12]
            track12_y = tracks_yz_12[track_id_12 // 10000]
            track12 = [track12_x, track12_y]

            hits12_x = hits_xz_12[track_id_12]
            hits12_y = hits_yz_12[track_id_12 // 10000]
            hits12 = list(hits12_x) + list(hits12_y)

            for track_id_34 in tracks_xz_34.keys():

                track34_x = tracks_xz_34[track_id_34]
                track34_y = tracks_yz_34[track_id_34 // 10000]
                track34 = [track34_x, track34_y]

                hits34_x = hits_xz_34[track_id_34]
                hits34_y = hits_yz_34[track_id_34 // 10000]
                hits34 = list(hits34_x) + list(hits34_y)



                [k_xz_12, b_xz_12] = track12[0]
                [k_yz_12, b_yz_12] = track12[1]

                [k_xz_34, b_xz_34] = track34[0]
                [k_yz_34, b_yz_34] = track34[1]

                dx, dy = get_dx_dy(track12, track34, zmagnet)

                true_label = get_true_label(hits12, hits34, all_hits)


                data.loc[len(data)] = [event_id,
                                       dx, dy,
                                       k_xz_12, b_xz_12,
                                       k_yz_12, b_yz_12,
                                       k_xz_34, b_xz_34,
                                       k_yz_34, b_yz_34,
                                       true_label,
                                       track_id_12, track_id_34]






    return data


def get_new_matched_tracks(reco_events12, reco_events34, match_tracks):
    """
    Match tracks reconstructed before and after the magnet.
    :param dict reco_events12: dictionary of the reconstructed tracks before the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :param dict reco_events34: dictionary of the reconstructed tracks after the magnet.
                               Key - event number,
                               value - [tracks_yz, linking_table_yz, tracks_xz, linking_table_xz].
    :return: dictionary of the matched tracks, where key - event number, value - list of the matched tracks ids;
             dictionary of charge values of the tracks, where key - event number, value - list of charge values;
             dictionary of inverse momentum values of the tracks, where key - event number, value - list of inverse momentum values;
             list of distances between the matched tracks on y;
             list of distances between the matched tracks on x.
    """

    dist_y = []
    dist_x = []
    maching_dict = {}
    charge_dict = {}
    pinv_dict = {}

    Bm = -0.75
    zmagnet = 3070.

    for event_id in reco_events12.keys():

        tracks_yz_12 = reco_events12[event_id][0]
        tracks_xz_12 = reco_events12[event_id][2]

        tracks_yz_34 = reco_events34[event_id][0]
        tracks_xz_34 = reco_events34[event_id][2]

        maching_dict[event_id] = []
        charge_dict[event_id] = []
        pinv_dict[event_id] = []

        for track_id_12, track_id_34 in match_tracks[event_id]:


            track12_x = tracks_xz_12[track_id_12]
            track12_y = tracks_yz_12[track_id_12 // 10000]
            track12 = [track12_x, track12_y]


            track34_x = tracks_xz_34[track_id_34]
            track34_y = tracks_yz_34[track_id_34 // 10000]
            track34 = [track34_x, track34_y]

            dx, dy = get_dx_dy(track12, track34, zmagnet)


            # distance
            maching_dict[event_id].append([track_id_12, track_id_34])
            dist_y.append(dy)
            dist_x.append(dx)

            # charge
            charge = get_charge(track12, track34)
            charge_dict[event_id].append(charge)

            # momentum inv
            pinv = get_pinv(track12, track34, Bm)
            pinv_dict[event_id].append(pinv)




    return charge_dict, pinv_dict, dist_y, dist_x
