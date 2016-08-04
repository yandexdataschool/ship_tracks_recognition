import numpy
import pandas
from sklearn.linear_model import LinearRegression
import itertools
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import scipy


class JustLinearRegression(object):

    def __init__(self):

        pass

    def fit(self, x, y, sample_weight=None):

        # self.slope, \
        # self.interept, \
        # self.rvalue, \
        # self.pvalue, \
        # self.stderr = scipy.stats.linregress(x, y)

        self.slope, self.interept = numpy.polyfit(x, y, 1)
        self.rvalue = numpy.corrcoef(x, y)[0, 1]
        self.stderr = numpy.std(numpy.abs(y - self.slope * x - self.interept))


class MultiLinearRegression(object):

    def __init__(self, n_tracks, 
                 min_samples=4,
                 subsample=1., 
                 x_unique=True, 
                 n_neighbors=10, 
                 step=0.05,  
                 track_classifier=None,
                 hit_classifier=None):

        self.min_samples = min_samples
        self.subsample = subsample
        self.x_unique = x_unique
        self.n_neighbors = n_neighbors
        self.step = step
        self.n_tracks = n_tracks
        self.track_classifier = track_classifier
        self.hit_classifier = hit_classifier
        self.unique_sorted_dists = []
        self.unique_sorted_indeces = []


        self.track_classification_data_ = {}
        self.hit_classification_data_ = {}


    def get_index_combinations(self, x, min_samples):

        index_combinations = []

        indeces = range(len(x))
        numpy.random.seed(42)

        if self.subsample > 1:

            counter = 0

            while counter < self.subsample:

                one_combination = numpy.random.choice(a=indeces, size=min_samples, replace=False)

                if self.x_unique:

                    x_one = x[list(one_combination)]
                    x_one_unique = numpy.unique(x_one)

                    if len(x_one_unique) == min_samples:

                        index_combinations.append(list(one_combination))
                        counter += 1

                else:

                    index_combinations.append(list(one_combination))
                    counter += 1

        else:

            for one_combination in itertools.combinations(indeces, min_samples):


                if self.subsample >= numpy.random.rand():

                    if self.x_unique:

                        x_one = x[list(one_combination)]
                        x_one_unique = numpy.unique(x_one)

                        if len(x_one_unique) == min_samples:

                            index_combinations.append(list(one_combination))

                    else:

                        index_combinations.append(list(one_combination))

        return index_combinations



    def fit_lines(self, x, y, sample_weight=None):

        indexes = self.get_index_combinations(x, self.min_samples)

        slopes = []
        intercepts = []
        rvalues = []
        stderrs = []

        for ind in indexes:

            ind = list(ind)

            x_step = x[ind]
            y_step = y[ind]
            if sample_weight != None:
                sample_weight_step = sample_weight[ind]
            else:
                sample_weight_step = None

            mlr = JustLinearRegression()
            mlr.fit(x_step, y_step, sample_weight_step)

            slopes += [mlr.slope]
            intercepts += [mlr.interept]
            rvalues += [mlr.rvalue]
            stderrs += [mlr.stderr]

        return numpy.array(indexes), numpy.array(slopes), numpy.array(intercepts), \
               numpy.array(rvalues), numpy.array(stderrs)


    def collect_track_classification_data(self, indexes, slopes, intercepts, rvalues, stderrs):

        self.track_classification_data_['indexes'] = indexes
        self.track_classification_data_['slopes'] = slopes
        self.track_classification_data_['intercepts'] = intercepts
        self.track_classification_data_['rvalues'] = rvalues
        self.track_classification_data_['stderrs'] = stderrs

    def track_classification(self, indexes, slopes, intercepts, rvalues, stderrs):


        self.collect_track_classification_data(indexes, slopes, intercepts, rvalues, stderrs)

        if self.track_classifier != None:

            X = numpy.concatenate(  (slopes.reshape(-1,1),
                                     intercepts.reshape(-1,1),
                                     rvalues.reshape(-1,1),
                                     stderrs.reshape(-1,1)), axis=1)
            y_predict = self.track_classifier.predict(X)

            return indexes[y_predict == 1], slopes[y_predict == 1], intercepts[y_predict == 1], \
                   rvalues[y_predict == 1], stderrs[y_predict == 1]

        else:

            return indexes, slopes, intercepts, rvalues, stderrs


    def get_track_classification_data(self, labels):

        data = pandas.DataFrame()

        indexes = self.track_classification_data_['indexes']
        slopes = self.track_classification_data_['slopes']
        intercepts = self.track_classification_data_['intercepts']
        rvalues = self.track_classification_data_['rvalues']
        stderrs = self.track_classification_data_['stderrs']

        unique_labels = [numpy.unique(labels[i]) for i in indexes]
        clf_labels = [(len(i) == 1)*(i[0] != -1)*1 for i in unique_labels]

        data['slope'] = slopes
        data['intercept'] = intercepts
        data['rvalue'] = numpy.nan_to_num(rvalues)
        data['stderr'] = numpy.nan_to_num(stderrs)
        data['label'] = clf_labels

        return data

    def knn_max_density(self, X, n_neighbors, step):

        # ss = StandardScaler()
        # ss.fit(X)
        # X_standart = ss.transform(X)
        #
        # passed_points_indeces = range(len(X_standart))
        # X_passed_standart = X_standart
        #
        # while len(X_passed_standart) > n_neighbors:
        #
        #     knn = NearestNeighbors(n_neighbors=n_neighbors, leaf_size=100)
        #     knn.fit(X_passed_standart)
        #     knn_dists, knn_indeces = knn.kneighbors()
        #
        #     knn_dists_mean = knn_dists.mean(axis=1)
        #
        #     n_points = max(1, int(step * len(X_passed_standart)))
        #     passed_points_indeces = knn_dists_mean.argsort()[:-n_points]
        #     knn_dists_mean.sort()
        #
        #     X_passed_standart = X_passed_standart[passed_points_indeces]
        #
        # X_passed = ss.inverse_transform(X_passed_standart)

        ss = StandardScaler()
        ss.fit(X)
        X_standart = ss.transform(X)

        passed_points_indeces = range(len(X_standart))
        X_passed_standart = X_standart


        n_neighbors = min(n_neighbors, len(X_passed_standart) - 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors, leaf_size=100)
        knn.fit(X_passed_standart)
        knn_dists, knn_indeces = knn.kneighbors()

        knn_dists_mean = knn_dists.mean(axis=1)

        max_dense_point = knn_dists_mean.argsort()[0]

        passed_points_indeces = list(knn_indeces[max_dense_point]) + [max_dense_point]

        X_passed_standart = X_passed_standart[passed_points_indeces]

        X_passed = ss.inverse_transform(X_passed_standart)

        return X_passed


    def get_track_params(self, X):

        ss = StandardScaler()
        ss.fit(X)

        transformed_tracks = ss.transform(X).mean(axis=0)
        tracks = ss.inverse_transform(transformed_tracks)
            
        return tracks, X.std(axis=0)
    
    def hit_clusterization(self, dists):

        max_model = None
        max_score = -1

        for n_clusters in [2, 3, 4]:

            km = KMeans(n_clusters=n_clusters)
            km.fit(dists.reshape(-1, 1))

            score = silhouette_score(dists.reshape(-1, 1), km.labels_)

            if score > max_score:

                max_score  = score
                max_model = km


        if max_score >= 0.7:

            min_cluster_id = max_model.cluster_centers_.reshape(-1).argsort()[0]
            indeces = numpy.arange(len(dists))[max_model.labels_ == min_cluster_id]

        else:

            indeces = numpy.arange(len(dists))
                
        return indeces

    def collect_hit_classification_data(self, indexes, dists, track, track_stderr):

        if self.hit_classification_data_.has_key('indexes'):

            self.hit_classification_data_['indexes'] += list(indexes)
            self.hit_classification_data_['dists'] += list(dists)
            self.hit_classification_data_['slopes'] += list([[track[0]]] * len(dists))
            self.hit_classification_data_['intercepts'] += list([[track[1]]] * len(dists))
            self.hit_classification_data_['slope_stderr'] += list([[track_stderr[0]]] * len(dists))
            self.hit_classification_data_['intercept_stderr'] += list([[track_stderr[1]]] * len(dists))
            self.hit_classification_data_['track_id'] += [self.hit_classification_data_['track_id'][-1] + 1] * len(dists)

        else:

            self.hit_classification_data_['indexes'] = list(indexes)
            self.hit_classification_data_['dists'] = list(dists)
            self.hit_classification_data_['slopes'] = list([[track[0]]] * len(dists))
            self.hit_classification_data_['intercepts'] = list([[track[1]]] * len(dists))
            self.hit_classification_data_['slope_stderr'] = list([[track_stderr[0]]] * len(dists))
            self.hit_classification_data_['intercept_stderr'] = list([[track_stderr[1]]] * len(dists))
            self.hit_classification_data_['track_id'] = [0] * len(dists)
    
    def hit_classification(self, indexes, dists, track, track_stderr):

        self.collect_hit_classification_data(indexes, dists, track, track_stderr)

        if self.hit_classifier != None:

            slopes = [[track[0]]] * len(dists)
            intercepts = [[track[1]]] * len(dists)
            slope_stderr = [[track_stderr[0]]] * len(dists)
            intercept_stderr = [[track_stderr[1]]] * len(dists)

            X = numpy.concatenate((dists.reshape(-1,1), slopes, intercepts, slope_stderr, intercept_stderr), axis=1)
            predict = self.hit_classifier.predict(X)

            return numpy.arange(len(dists))[predict == 1]

        else:

            track_indeces = self.hit_clusterization(dists)

            return track_indeces

    def get_hit_classification_data(self, labels):

        data = pandas.DataFrame()

        indexes = self.hit_classification_data_['indexes']
        dists = numpy.array(self.hit_classification_data_['dists'])
        slopes = numpy.array(self.hit_classification_data_['slopes'])
        intercepts = numpy.array(self.hit_classification_data_['intercepts'])
        slope_stderr = numpy.array(self.hit_classification_data_['slope_stderr'])
        intercept_stderr = numpy.array(self.hit_classification_data_['intercept_stderr'])
        track_ids = numpy.array(self.hit_classification_data_['track_id'])

        hit_labels = labels[indexes]

        clf_labels = numpy.zeros(len(indexes))


        for track_id in numpy.unique(track_ids):

            dists_track = dists[track_ids == track_id]
            hit_labels_track = hit_labels[track_ids == track_id]

            unique = numpy.unique(hit_labels_track)
            dist_means = numpy.array([dists_track[hit_labels_track == i].mean() for i in unique])

            true_track_id = unique[dist_means == dist_means.min()]

            clf_labels[(track_ids == track_id)*(hit_labels == true_track_id)*(hit_labels != -1)] = 1

        data['dist'] = dists
        data['slope'] = slopes[:, 0]
        data['intercept'] = intercepts[:, 0]
        data['slope_stderr'] = slope_stderr[:, 0]
        data['intercept_stderr'] = intercept_stderr[:, 0]
        data['label'] = clf_labels

        return data


    
    def select_track_hits(self, x, y, track, one_track_stderr):
        
        dists = numpy.abs(y - (track[:-1] * x.reshape(-1, 1)).sum(axis=1) - track[-1])
        sorted_dists_indeces = dists.argsort()
        
        indeces = numpy.array(range(len(x)))

        sorted_dists = dists[sorted_dists_indeces]
        sorted_x = x[sorted_dists_indeces]
        sorted_y = y[sorted_dists_indeces]
        sorted_indeces = indeces[sorted_dists_indeces]

        unique, index = numpy.unique(sorted_x, return_index=True)

        unique_dists = sorted_dists[index]
        unique_X = sorted_x[index]
        unique_y = sorted_y[index]
        unique_indeces = sorted_indeces[index]

        sorted_dists_indeces = unique_dists.argsort()

        unique_sorted_dists = unique_dists[sorted_dists_indeces]
        unique_sorted_x = unique_X[sorted_dists_indeces]
        unique_sorted_y = unique_y[sorted_dists_indeces]
        unique_sorted_indeces = unique_indeces[sorted_dists_indeces]
        
        self.unique_sorted_dists.append(unique_sorted_dists)
        self.unique_sorted_indeces.append(unique_sorted_indeces)
        
        track_indeces = self.hit_classification(unique_sorted_indeces, unique_sorted_dists, track, one_track_stderr)
        
        return unique_sorted_x[track_indeces], unique_sorted_y[track_indeces], unique_sorted_indeces[track_indeces]
        
        
        
    def find_one_track(self, slopes, intercepts):

        track_candidates = numpy.concatenate(  (slopes.reshape(-1, 1),
                                               intercepts.reshape(-1, 1)), axis=1)
        
        n_neighbors = self.n_neighbors
        step = self.step

        track_candidates_dense = self.knn_max_density(track_candidates, n_neighbors, step)

        tracks, tracks_stderr = self.get_track_params(track_candidates_dense)
        
        return tracks, tracks_stderr
    
    def _isAinB(self, A, B):
    
        return len(set(A) & set(B)) != 0
    
    def fit_all_tracks(self, x, y, indexes, slopes, intercepts):
        
        tracks = []
        tracks_labels = -1 * numpy.ones(len(x))

        slopes_not_used, intercepts_not_used = slopes, intercepts
        used = []

        for num in range(self.n_tracks):

            one_track, one_track_stderr = self.find_one_track(slopes_not_used, intercepts_not_used)


            one_track_x, one_track_y, one_track_indexes = self.select_track_hits(x, y, one_track, one_track_stderr)



            tracks_labels[one_track_indexes] = num
            tracks.append(one_track)


            used += list(one_track_indexes)

            selection = numpy.array([not self._isAinB(i, used) for i in indexes])
            slopes_not_used = slopes[selection]
            intercepts_not_used = intercepts[selection]
                
        return tracks, tracks_labels
                
    
    def fit(self, x, y, sample_weight=None):

        indexes, slopes, intercepts, rvalues, stderrs = self.fit_lines(x, y, sample_weight)

        indexes, slopes, intercepts, rvalues, stderrs = \
            self.track_classification(indexes, slopes, intercepts, rvalues, stderrs)

        self.indexes_ = indexes
        self.slopes_ = slopes
        self.intercepts_ = intercepts
        self.rvalues_ = rvalues
        self.stderrs_ = stderrs

        
        tracks, tracks_labels = self.fit_all_tracks(x, y, indexes, slopes, intercepts)


        self.tracks_params_ = numpy.array(tracks)
        self.labels_ = tracks_labels


#################################################################################
# Data Collection
#################################################################################

from copy import copy, deepcopy
import numpy
import pandas

class TracksReconstruction2DWithDataCollection(object):

    def __init__(self, model_y, model_stereo):
        """
        This is realization of the reconstruction scheme which uses two 2D projections to reconstruct a 3D track.
        :param model_y: model for the tracks reconstruction in y-z plane.
        :param model_stereo: model for the tracks reconstruction in x-z plane.
        :return:
        """

        self.model_y = deepcopy(model_y)
        self.model_stereo = deepcopy(model_stereo)

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

        self.labels_ = -1. * numpy.ones(len(event))
        self.tracks_params_ = []

        # Tracks Reconstruction in Y-view
        event_y = event[event.IsStereo == 0]
        mask_y = event.IsStereo.values == 0

        x_y = event_y.Wz1.values
        y_y = event_y.Wy1.values
        true_labels_y = event_y.Label.values

        if sample_weight != None:
            sample_weight_y = sample_weight[mask_y == 1]
        else:
            sample_weight_y = None


        self.model_y.fit(x_y, y_y, sample_weight_y)

        self.track_clf_data_y_ = self.model_y.get_track_classification_data(true_labels_y)
        self.hit_clf_data_y_ = self.model_y.get_hit_classification_data(true_labels_y)


        labels_y = self.model_y.labels_
        tracks_params_y = self.model_y.tracks_params_

        self.labels_[mask_y] = labels_y

        # Tracks Reconstruction in Stereo_views
        event_stereo = event[event.IsStereo == 1]
        used = numpy.zeros(len(event_stereo))
        mask_stereo = event.IsStereo.values == 1

        self.track_clf_data_stereo_ = pandas.DataFrame()
        self.hit_clf_data_stereo_ = pandas.DataFrame()

        for track_id, one_track_y in enumerate(tracks_params_y):

            plane_k, plane_b = one_track_y
            x_stereo, y_stereo = self.get_xz(plane_k, plane_b, event_stereo)
            true_labels_stereo = event_stereo.Label.values

            if sample_weight != None:
                sample_weight_stereo = sample_weight[mask_stereo == 1][used==0]
            else:
                sample_weight_stereo = None

            new_model_stereo = deepcopy(self.model_stereo)
            new_model_stereo.fit(x_stereo[used==0], y_stereo[used==0], sample_weight_stereo)

            self.track_clf_data_stereo_ = self.track_clf_data_stereo_.append(new_model_stereo.get_track_classification_data(true_labels_stereo[used==0]))
            self.hit_clf_data_stereo_ = self.hit_clf_data_stereo_.append(new_model_stereo.get_hit_classification_data(true_labels_stereo[used==0]))

            labels_stereo = -1. * numpy.ones(len(event_stereo))
            labels_stereo[used==0] = new_model_stereo.labels_
            tracks_params_stereo = new_model_stereo.tracks_params_

            unique, counts = numpy.unique(labels_stereo[labels_stereo != -1], return_counts=True)
            if len(unique) != 0:
                max_hits_track_id = unique[counts == counts.max()][0]
                one_track_stereo = tracks_params_stereo[max_hits_track_id]
            else:
                max_hits_track_id = -999.
                one_track_stereo = []

            used[labels_stereo == max_hits_track_id] = 1

            self.labels_[mask_stereo] = track_id * (labels_stereo == max_hits_track_id) + \
            self.labels_[mask_stereo] * (labels_stereo != max_hits_track_id)


            self.tracks_params_.append([one_track_y, one_track_stereo])

        self.tracks_params_ = numpy.array(self.tracks_params_)


class DataCollection(object):

    def __init__(self, model_y, model_stereo):

        self.model_y = deepcopy(model_y)
        self.model_stereo = deepcopy(model_stereo)


    def one_event_clf_train_data(self, event):

        # Get an event
        event12 = event[(event.StatNb == 1) + (event.StatNb == 2)]
        event34 = event[(event.StatNb == 3) + (event.StatNb == 4)]

        weights12 = None# 1. / numpy.sqrt(event12.dist2Wire.values**2 + 0.01**2)
        weights34 = None# 1. / numpy.sqrt(event34.dist2Wire.values**2 + 0.01**2)

        # Select model for the tracks reconstruction
        stm_y = deepcopy(self.model_y)
        stm_stereo = deepcopy(self.model_stereo)

        # Tracks reconstruction before the magnet
        tr2d12 = TracksReconstruction2DWithDataCollection(model_y=stm_y, model_stereo=stm_stereo)
        tr2d12.fit(event12, weights12)

        track_clf_data_12y = tr2d12.track_clf_data_y_
        hit_clf_data_12y = tr2d12.hit_clf_data_y_
        track_clf_data_12stereo = tr2d12.track_clf_data_stereo_
        hit_clf_data_12stereo = tr2d12.hit_clf_data_stereo_


        # Tracks reconstruction after the magnet
        tr2d34 = TracksReconstruction2DWithDataCollection(model_y=stm_y, model_stereo=stm_stereo)
        tr2d34.fit(event34, weights34)

        track_clf_data_34y = tr2d34.model_y.get_track_classification_data(event34.Label.values[event34.IsStereo.values == 0])

        track_clf_data_34y = tr2d34.track_clf_data_y_
        hit_clf_data_34y = tr2d34.hit_clf_data_y_
        track_clf_data_34stereo = tr2d34.track_clf_data_stereo_
        hit_clf_data_34stereo = tr2d34.hit_clf_data_stereo_


        hit_clf_data_y = pandas.concat([hit_clf_data_12y, hit_clf_data_34y], axis=0)
        track_clf_data_y = pandas.concat([track_clf_data_12y, track_clf_data_34y], axis=0)
        hit_clf_data_stereo = pandas.concat([hit_clf_data_12stereo, hit_clf_data_34stereo], axis=0)
        track_clf_data_stereo = pandas.concat([track_clf_data_12stereo, track_clf_data_34stereo], axis=0)

        return track_clf_data_y, hit_clf_data_y, track_clf_data_stereo, hit_clf_data_stereo

    def clf_train_data(self, data, event_ids):

        track_clf_data_y = pandas.DataFrame()
        hit_clf_data_y = pandas.DataFrame()
        track_clf_data_stereo = pandas.DataFrame()
        hit_clf_data_stereo = pandas.DataFrame()

        for event_id in event_ids:

            event = data[data.EventID == event_id]

            one_track_clf_data_y, \
            one_hit_clf_data_y, \
            one_track_clf_data_stereo, \
            one_hit_clf_data_stereo = self.one_event_clf_train_data(event)

            track_clf_data_y = track_clf_data_y.append(one_track_clf_data_y)
            hit_clf_data_y = hit_clf_data_y.append(one_hit_clf_data_y)
            track_clf_data_stereo = track_clf_data_stereo.append(one_track_clf_data_stereo)
            hit_clf_data_stereo = hit_clf_data_stereo.append(one_hit_clf_data_stereo)

        return track_clf_data_y, hit_clf_data_y, track_clf_data_stereo, hit_clf_data_stereo