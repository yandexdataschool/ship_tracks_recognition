__author__ = 'mikhail91'

from copy import deepcopy
from multiprocessing import Pool
from utils import merge_dicts

import numpy
import pandas

from sklearn.cross_validation import train_test_split

from rep.metaml import FoldingClassifier
from rep.estimators import XGBoostClassifier
from rep.estimators import XGBoostClassifier

from reconstruction import TracksReconstruction2D
from combination import Combinator, SuperCombinator


from metrics import TracksReconstractionMetrics, CombinatorQuality

def tracks_reconstruction(params):

    event_id, data, model_y, model_stereo = params

    # Get an event
    event = data[data.EventID == event_id]
    event12 = event[(event.StatNb == 1) + (event.StatNb == 2)]
    event34 = event[(event.StatNb == 3) + (event.StatNb == 4)]

    weights12 = None#1. / numpy.sqrt(event12.dist2Wire.values**2 + 0.01**2)
    weights34 = None#1. / numpy.sqrt(event34.dist2Wire.values**2 + 0.01**2)

    # Select model for the tracks reconstruction
    #model_y = Baline(n_max_hits=16, n_min_hits=7, window_width=0.7)
    #model_stereo = Baline(n_max_hits=16, n_min_hits=6, window_width=15)


    # Tracks reconstruction before the magnet
    tr2d12 = TracksReconstruction2D(model_y=model_y, model_stereo=model_stereo)
    tr2d12.fit(event12, weights12)

    labels12 = tr2d12.labels_
    tracks_params12 = tr2d12.tracks_params_

    # Tracks reconstruction after the magnet
    tr2d34 = TracksReconstruction2D(model_y=model_y, model_stereo=model_stereo)
    tr2d34.fit(event34, weights34)

    labels34 = tr2d34.labels_
    tracks_params34 = tr2d34.tracks_params_

    one_track = {}
    one_track[event_id] = {'labels12':labels12, 'params12':tracks_params12,
                           'labels34':labels34, 'params34':tracks_params34}

    return one_track


def get_eff_value(params):

    event_id, data, tracks, clf = params

    # Get an event
    event = data[data.EventID == event_id]
    event12 = event[(event.StatNb == 1) + (event.StatNb == 2)]
    event34 = event[(event.StatNb == 3) + (event.StatNb == 4)]


    labels12 = tracks[event_id]['labels12']
    tracks_params12 = tracks[event_id]['params12']
    labels34 = tracks[event_id]['labels34']
    tracks_params34 = tracks[event_id]['params34']


    # Quality metrics of the reconstruction
    trm12 = TracksReconstractionMetrics(0.2, n_tracks=2)
    trm12.fit(labels12, event12)


    # Quality metrics of the reconstruction
    trm34 = TracksReconstractionMetrics(0.2, n_tracks=2)
    trm34.fit(labels34, event34)


    # Combination of the tracks before and after the magnet
    if clf == None:
        comb = Combinator()
    else:
        comb = SuperCombinator(classifier=clf)
    comb.combine(tracks_params12, tracks_params34)


    # Quality of the combination
    cq = CombinatorQuality()
    cq.fit(labels12, labels34, comb.tracks_combinations_, comb.charges_, comb.inv_momentums_, event12, event34)


    # Overall quality
    line = numpy.array([1,
                         (trm12.recognition_efficiency_y_ == 1)*1,
                         (trm12.recognition_efficiency_stereo_ >= 1.)*1,
                         (trm12.recognition_efficiency_ == 1)*1,
                         (trm34.recognition_efficiency_y_ == 1)*1,
                         (trm34.recognition_efficiency_stereo_ >= 1.)*1,
                         (trm34.recognition_efficiency_ == 1)*1,
                         (cq.n_combined_ >= 2)*1,
                         (cq.reco_eff_ == 1)*1]).cumprod()

    return line[-1]


class GridSearch2D(object):

    def __init__(self, model_y, model_stereo, params_y, params_stereo, processes, train_size=None):

        self.model_y = model_y
        self.model_stereo = model_stereo
        self.params_y = params_y
        self.params_stereo = params_stereo
        self.processes = processes
        self.train_size = train_size

    def params_splitter(self, key_id, params, params_one, params_one_list):


        if key_id >= len(params.keys()):

            params_one_list.append(params_one)
            return

        for val in params[params.keys()[key_id]]:

            params_one_new = params_one.copy()
            params_one_new[params.keys()[key_id]] = val

            self.params_splitter(key_id + 1, params, params_one_new, params_one_list)

    def get_best_eff(self, results):

        effs = numpy.array(results.keys())
        best_eff = effs.max()

        return best_eff, results[best_eff]


    def fit_one(self, data, model_y, model_stereo):


        event_ids = numpy.unique(data.EventID.values)

        if self.train_size != None:
            event_ids_train, event_ids_test= train_test_split(event_ids, train_size=self.train_size, random_state=42)
        else:
            event_ids_test = event_ids

        # fit train tracks
        if self.train_size != None:

            tracks_train = {}

            p = Pool(self.processes)
            results_train = p.map(tracks_reconstruction, zip(event_ids_train,
                                                       [data]*len(event_ids_train),
                                                       [model_y]*len(event_ids_train),
                                                       [model_stereo]*len(event_ids_train)))
            tracks_train = merge_dicts(results_train)

        # train clf
        if self.train_size != None:

            sc = SuperCombinator()

            combination_data = sc.data_collection(tracks_train, data)

            X_data = combination_data[combination_data.columns[:-1]].values
            y_data = combination_data.label.values

            xgb_base = XGBoostClassifier(n_estimators=1000, colsample=0.7, eta=0.01, nthreads=1,
                             subsample=0.7, max_depth=8)
            folding = FoldingClassifier(xgb_base, n_folds=10, random_state=11)
            folding.fit(X_data, y_data)

            clf = folding.estimators[0]

        else:
            clf = None



        # fit test tracks
        tracks_test = {}

        p = Pool(self.processes)
        results_test = p.map(tracks_reconstruction, zip(event_ids_test,
                                                   [data]*len(event_ids_test),
                                                   [model_y]*len(event_ids_test),
                                                   [model_stereo]*len(event_ids_test)))
        tracks_test = merge_dicts(results_test)


        # quality
        p = Pool(self.processes)
        effs = p.map(get_eff_value, zip(event_ids_test,
                                       [data]*len(event_ids_test),
                                       [tracks_test]*len(event_ids_test),
                                       [clf]*len(event_ids_test)))

        eff = 100. * numpy.array(effs).sum() / len(effs)

        return eff

    def fit(self, data):

        results = {}

        params_y_list = []
        self.params_splitter(0, self.params_y, {}, params_y_list)

        params_stereo_list = []
        self.params_splitter(0, self.params_stereo, {}, params_stereo_list)

        for params_y_one in params_y_list:
            for params_stereo_one in params_stereo_list:

                model_y = deepcopy(self.model_y)
                model_y.__init__(**params_y_one)

                model_stereo = deepcopy(self.model_stereo)
                model_stereo.__init__(**params_stereo_one)

                eff = self.fit_one(data, model_y, model_stereo)

                results[eff] = {'params_y': params_y_one, 'params_stereo': params_stereo_one}

        self.results_ = results

        self.best_eff_, self.best_params_ = self.get_best_eff(results)
