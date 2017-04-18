__author__ = 'mikhail91'

import numpy



def get_tracks_params(reconstructible_events, stations, plane, all_hits):

    params = {}

    for event_id in reconstructible_events.keys():

        event = all_hits[all_hits['event'] == event_id]
        event = event[(event.StatNb.values == stations[0]) + (event.StatNb.values == stations[1])]

        ks = {}
        bs = {}
        covs = {}

        for track_id in [2, 3]:

            track = event[event.TrackID == track_id]
            if len(track) == 0:
                continue

            if plane=='zy':
                (k, b), cov = numpy.polyfit(track.Z.values, track.Y.values, 1, cov=True)
            elif plane=='zx':
                (k, b), cov = numpy.polyfit(track.Z.values, track.X.values, 1, cov=True)

            if (numpy.abs(cov) == numpy.inf).sum() > 0:
                continue

            ks[track_id] = k
            bs[track_id] = b
            covs[track_id] = cov

        params[event_id] = [ks, bs, covs]

    return params


def get_params_distributions(params):

    pair = numpy.array([len(i[0].keys()) for i in params.values()])
    res = numpy.array(params.values())

    ks = numpy.array([i[0][2] for i in res[pair == 2]] + [i[0][3] for i in res[pair == 2]])
    dks = numpy.abs([i[0][2] - i[0][3] for i in res[pair == 2]])

    bs = numpy.array([i[1][2] for i in res[pair == 2]] + [i[1][3] for i in res[pair == 2]])
    dbs = numpy.abs([i[1][2] - i[1][3] for i in res[pair == 2]])

    k_errs = numpy.array([i[2][2][0,0] for i in res[pair == 2]] + [i[2][3][0,0] for i in res[pair == 2]])
    b_errs = numpy.array([i[2][2][1,1] for i in res[pair == 2]] + [i[2][3][1,1] for i in res[pair == 2]])

    return ks, dks, bs, dbs, k_errs, b_errs


def merge_dicts(dicts):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def get_sample_weight(event):

    stat_views = event.StatNb.values * 100 + event.ViewNb.values * 10 + event.PlaneNb.values
    unique, counts = numpy.unique(stat_views, return_counts=True)

    sample_weight = numpy.zeros(len(event))

    for val, count in zip(unique, counts):

        sample_weight += (stat_views == val) * 1. / count

    return sample_weight


import matplotlib.pyplot as plt

def plot_event(event_id, data, tracks):

    event = data[data.EventID == event_id]
    track = tracks[event_id]

    event12 = event[(event.StatNb == 1) + (event.StatNb == 2)]
    event34 = event[(event.StatNb == 3) + (event.StatNb == 4)]

    track12 = track['params12']
    track34 = track['params34']

    plt.figure(figsize=(14, 10))

    plt.subplot(2,2,1)
    plt.scatter(event12.Z.values, event12.Y.values)

    for track_id in range(len(track12)):

        plt.plot(event12.Z.values, event12.Z.values * track12[track_id][0][0] + track12[track_id][0][1])

    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.title('Stations 1&2')

    plt.subplot(2,2,2)
    plt.scatter(event12.Z.values, event12.X.values)

    for track_id in range(len(track12)):

        if len(track12[track_id][1]) == 0:
            continue

        plt.plot(event12.Z.values, event12.Z.values * track12[track_id][1][0] + track12[track_id][1][1])

    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title('Stations 1&2')

    plt.subplot(2,2,3)
    plt.scatter(event34.Z.values, event34.Y.values)

    for track_id in range(len(track34)):

        plt.plot(event34.Z.values, event34.Z.values * track34[track_id][0][0] + track34[track_id][0][1])

    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.title('Stations 3&4')

    plt.subplot(2,2,4)
    plt.scatter(event34.Z.values, event34.X.values)

    for track_id in range(len(track34)):

        if len(track34[track_id][1]) == 0:
            continue

        plt.plot(event34.Z.values, event34.Z.values * track34[track_id][1][0] + track34[track_id][1][1])

    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title('Stations 3&4')
    plt.show()



################# Efficiency per track #############################
import pandas
from metrics import TracksReconstractionMetrics

def get_effs_per_track_and_p_station(tracks, data, event_ids, stations='12'):

    tracks_eff_before = pandas.DataFrame(columns=['Eff', 'Momentum'])

    for event_id in event_ids:

        if stations == '12':

            labels = tracks[event_id]['labels12']

            event = data[data.EventID == event_id]
            event = event[(event.StatNb == 1) + (event.StatNb == 2)]

        elif stations == '34':

            labels = tracks[event_id]['labels34']

            event = data[data.EventID == event_id]
            event = event[(event.StatNb == 3) + (event.StatNb == 4)]

        true_labels = event.Label.values

        trm = TracksReconstractionMetrics(0.2)
        trm.fit(labels, event)

        eff = trm.efficiencies_


        for num, lab in enumerate(numpy.unique(labels[labels != -1])):

            valid_labs = true_labels[labels == lab]
            unique, counts = numpy.unique(valid_labs[valid_labs!=-1], return_counts=True)

            if len(unique) == 0:
                continue

            true_lab = unique[counts == counts.max()][0]

            track = event[true_labels == true_lab]
            p_one = numpy.sqrt(track.Px.values**2+track.Py.values**2+track.Pz.values**2).mean()

            eff_one = eff[num]

            tracks_eff_before.loc[len(tracks_eff_before)] = [eff_one, p_one]

    return tracks_eff_before

def get_effs_per_track_and_p(tracks, data, event_ids):

    tracks_eff_before12 = get_effs_per_track_and_p_station(tracks, data, event_ids, stations='12')
    tracks_eff_before34 = get_effs_per_track_and_p_station(tracks, data, event_ids, stations='34')

    return pandas.concat([tracks_eff_before12, tracks_eff_before34], axis=0)


def get_bins(x, y, bins, x_min, x_max):

    step = 1. * ( x_max - x_min ) / bins
    edges = [x_min + i * step for i in range(0, bins+1)]

    y_means = []
    y_err = []
    x_err = []
    x_means = []

    for i in range(0, len(edges)-1):

        left = edges[i]
        right = edges[i+1]

        if i == len(edges)-2:

            y_bin = y[(x >= left) * (x <= right)]

        else:

            y_bin = y[(x >= left) * (x < right)]


        y_means.append(y_bin.mean())
        y_err.append(1. * y_bin.std() / (len(y_bin) + 0.001))
        x_means.append(0.5*(left + right))
        x_err.append(0.5*(-left + right))

    return x_means, y_means, x_err, y_err


def plot_efficiency_per_track(tracks, data, event_ids, bins):

    tracks_eff = get_effs_per_track_and_p(tracks, data, event_ids)

    x_means, y_means, x_err, y_err = get_bins(tracks_eff.Momentum.values,
                                              tracks_eff.Eff.values,
                                              bins,
                                              tracks_eff.Momentum.values.min(),
                                              tracks_eff.Momentum.values.max())

    plt.figure(figsize=(10, 7))
    plt.errorbar(x_means, y_means, xerr=x_err, yerr=y_err, fmt='none')
    plt.scatter(x_means, y_means, linewidth=0, color='r')
    plt.ylim(0.0, 1.05)
    plt.xlim(0, x_means[-1]+x_err[-1]+1)
    plt.xlabel('Particle Momentum', size=15)
    plt.ylabel('Efficiency per track', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)







