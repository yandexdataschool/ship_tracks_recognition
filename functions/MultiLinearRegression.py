import numpy
import pandas
from sklearn.linear_model import LinearRegression
import itertools, random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


class JustLinearRegression(object):

    def __init__(self):

        self.regressor = None # regressor's class

    def fit(self, X, y):

        qlr = LinearRegression()
        qlr.fit(X, y)

        self.regressor = qlr

        if len(X) > 2:
            score = numpy.sqrt(1. * (numpy.abs(qlr.predict(X) - y)**2).sum() / (len(X) - 2))
        else:
            score = numpy.sqrt(1. * (numpy.abs(qlr.predict(X) - y)**2).sum() / (len(X) - 1))

        return score, qlr


    def predict(self, X):

        return self.regressor.predict(X)


class MultiLinearRegression(object):

    def __init__(self, n_tracks, 
                 n=4, 
                 subsample=1., 
                 x_unique=True, 
                 n_neighbors=10, 
                 step=0.05,  
                 track_classifier=None,
                 hit_classifier=None):

        self.n = n
        self.subsample = subsample
        self.x_unique = x_unique
        self.n_neighbors = n_neighbors
        self.step = step
        self.n_tracks = n_tracks
        self.track_classifier = track_classifier
        self.hit_classifier = hit_classifier
        self.unique_sorted_dists = []
        self.unique_sorted_indeces = []


    def get_index_combinations(self, X, n):

        index_combinations = []

        indeces = range(len(X))
        random.seed = 42

        for one_combination in itertools.combinations(indeces, n):
            
            if self.subsample >= random.random():

                if self.x_unique:

                    X_one = X[list(one_combination)]
                    X_one_unique = numpy.unique(X_one)

                    if len(X_one_unique) == n:

                        index_combinations.append(list(one_combination))

                else:

                    index_combinations.append(list(one_combination))

        return index_combinations



    def fit_lines(self, X, y):

        scores_list = []
        lrs_list = []

        indexes_list = self.get_index_combinations(X, self.n)

        for ind in indexes_list:

            ind = list(ind)

            X_step = X[ind]
            y_step = y[ind]

            mlr = JustLinearRegression()
            score, lr = mlr.fit(X_step, y_step)
            
            scores_list.append(score)
            lrs_list.append(lr)

        return numpy.array(scores_list), numpy.array(lrs_list), numpy.array(indexes_list)
    
    def knn_max_density(self, X, n_neighbors, step):

        ss = StandardScaler()
        ss.fit(X)
        X_standart = ss.transform(X)

        passed_points_indeces = range(len(X_standart))
        X_passed_standart = X_standart

        while len(X_passed_standart) > n_neighbors:

            knn = NearestNeighbors(n_neighbors=n_neighbors, leaf_size=100)
            knn.fit(X_passed_standart)
            knn_dists, knn_indeces = knn.kneighbors()

            knn_dists_mean = knn_dists.mean(axis=1)

            n_points = max(1, int(step * len(X_passed_standart)))
            passed_points_indeces = knn_dists_mean.argsort()[:-n_points]
            knn_dists_mean.sort()

            X_passed_standart = X_passed_standart[passed_points_indeces]
            
        X_passed = ss.inverse_transform(X_passed_standart)

        return X_passed


    def get_tracks(self, X, n_tracks):

        ss = StandardScaler()
        ss.fit(X)

        transformed_tracks = ss.transform(X).mean(axis=0)
        tracks = ss.inverse_transform(transformed_tracks)
            
        return tracks
    
    def _select_array(self, array):

        max_model = None
        max_score = -1

        for n_clusters in [2, 3, 4]:

            km = KMeans(n_clusters=3)
            km.fit(array.reshape(-1, 1))

            score = silhouette_score(array.reshape(-1, 1), km.labels_)

            if score > max_score:

                max_score  = score
                max_model = km


        if max_score >= 0.7:

            min_cluster_id = max_model.cluster_centers_.reshape(-1).argsort()[0]
            indeces = numpy.arange(len(array))[max_model.labels_ == min_cluster_id]

        else:

            indeces = numpy.arange(len(array))
                
        return indeces
    
    def _hits_classification(self, dists, track):
        
        ks = [[track[0]]] * len(dists)
        bs = [[track[1]]] * len(dists)
        
        X = numpy.concatenate((dists.reshape(-1,1), ks, bs), axis=1)
        predict = self.hit_classifier.predict(X)
        
        return numpy.arange(len(dists))[predict == 1]
    
    def get_points(self, X, y, track):
        
        dists = numpy.abs(y - (track[:-1] * X).sum(axis=1) - track[-1])
        sorted_dists_indeces = dists.argsort()
        
        indeces = numpy.array(range(len(X)))

        sorted_dists = dists[sorted_dists_indeces]
        sorted_X = X[sorted_dists_indeces]
        sorted_y = y[sorted_dists_indeces]
        sorted_indeces = indeces[sorted_dists_indeces]

        unique, index = numpy.unique(sorted_X, return_index=True)

        unique_dists = sorted_dists[index]
        unique_X = sorted_X[index]
        unique_y = sorted_y[index]
        unique_indeces = sorted_indeces[index]

        sorted_dists_indeces = unique_dists.argsort()

        unique_sorted_dists = unique_dists[sorted_dists_indeces]
        unique_sorted_X = unique_X[sorted_dists_indeces]
        unique_sorted_y = unique_y[sorted_dists_indeces]
        unique_sorted_indeces = unique_indeces[sorted_dists_indeces]
        
        self.unique_sorted_dists.append(unique_sorted_dists)
        self.unique_sorted_indeces.append(unique_sorted_indeces)
        
        if self.hit_classifier == None:
        
            track_indeces = self._select_array(unique_sorted_dists)
            
        else:
            
            track_indeces = self._hits_classification(unique_sorted_dists, track)
        
        return unique_sorted_X[track_indeces], unique_sorted_y[track_indeces], unique_sorted_indeces[track_indeces]
        
        
        
    def _fit_satellite(self, scores, lrs):
        
        ks = numpy.array([lr.coef_ for lr in lrs])
        bs = numpy.array([lr.intercept_ for lr in lrs]).reshape((-1,1))
        
        track_candidates = numpy.concatenate((ks, bs), axis=1)
        
        n_neighbors = self.n_neighbors
        step = self.step
        track_candidates_dense = self.knn_max_density(track_candidates, n_neighbors, step)
        
        
        tracks = self.get_tracks(track_candidates_dense, self.n_tracks)
        
        return tracks
    
    def _classification(self, scores, lrs, indeces):
        
        ks = numpy.array([lr.coef_ for lr in lrs])
        bs = numpy.array([lr.intercept_ for lr in lrs]).reshape((-1,1))
        
        X = numpy.concatenate((scores.reshape(-1,1), ks, bs.reshape(-1,1)), axis=1)
        y_predict = self.track_classifier.predict(X)
        
        return scores[y_predict == 1], lrs[y_predict == 1], indeces[y_predict == 1]
    
    def _isAinB(self, A, B):
    
        return len(set(A) & set(B)) != 0
    
    def _get_all_tracks(self, X, y, scores, lrs, indeces, unique_index):
        
        tracks = []
        tracks_labels = -1 * numpy.ones(len(X))
        
        self.qw = []
        
        if unique_index == None:
            
            scores_curr = scores.copy()
            lrs_curr = lrs.copy()
            looked_inds = []
            
            for num in range(self.n_tracks):
                
                one_track = self._fit_satellite(scores_curr, lrs_curr)
                one_track_X, one_track_y, one_track_indeces = self.get_points(X, y, one_track)
                tracks_labels[one_track_indeces] = num
                tracks.append(one_track)
                
                self.qw.append(one_track_indeces)
                
                looked_inds += list(one_track_indeces)
                
                selection = numpy.array([not self._isAinB(i, looked_inds) for i in indeces])
                scores_curr = scores[selection]
                lrs_curr = lrs[selection]
                
        return tracks, tracks_labels
                
    
    def fit(self, X, y, unique_index=None):
        
        scores, lrs, indeces = self.fit_lines(X, y)
        
        if self.track_classifier != None:
            
            scores, lrs, indeces = self._classification(scores, lrs, indeces)
            self.scores_ = scores
            self.lrs_ = lrs
            self.indeces_ = indeces
        
            tracks, tracks_labels = self._get_all_tracks(X, y, scores, lrs, indeces, unique_index)
        
        else:
            
            self.scores_ = scores
            self.lrs_ = lrs
            self.indeces_ = indeces
        
            tracks, tracks_labels = self._get_all_tracks(X, y, scores, lrs, indeces, unique_index)
            
        
        self.tracks_ = tracks
        self.labels_ = tracks_labels


    def predict(self, X):
        
        predictions = []
        
        for track in self.tracks_:
            
            y_predict = (track[:-1] * X).sum(axis=1) + track[-1]
            predictions.append(y_predict)
            
        return numpy.array(predictions)