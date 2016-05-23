__author__ = 'mikhail91'

from ReconstructibleTracks import get_reconstractible_tracks
import numpy

def get_reconstractible_events(event_ids, all_hits, all_mctracks, all_velo_points):

    reconstructible_events = {}

    for event_id in event_ids:

        tracks = get_reconstractible_tracks(event_id, all_hits, all_mctracks, all_velo_points)

        if tracks != []:
            reconstructible_events[event_id] = tracks

    return reconstructible_events


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





