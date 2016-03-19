__author__ = 'mikhail91'

import copy
import numpy

class OneVsRestClassifier(object):

    def __init__(self, estimator):
        """
        One vs Rest classifier based on an estimator.
        :param estimator: estimator with fit, predict, predict_proba mathods.
        :return:
        """

        self.estimator = estimator
        self.estimators_ = {}
        self.classes = []

    def fit(self, X, y, sample_weight=None):
        """
        Fit the classifier.
        :param X: the estimator's format, data
        :param y: array, shape = [n_samples], labels of classes [0, 1, 2, ..., n_classes - 1]
        :param sample_weight: None, 'balanced' or array, shape = [n_samples], sample weights.
        If 'balanced' sum of weights of positive and negative classes will be equal.
        :return:
        """

        self.classes = numpy.unique(y)

        for one_class in self.classes:

            y_class = (y == one_class) * 1.

            estimator = copy.copy(self.estimator)

            if sample_weight is None:

                estimator.fit(X, y_class)

            elif sample_weight == 'balanced':

                weights = (y == one_class) * len(y) / ((y == one_class).sum()) + \
                          (y != one_class) * len(y) / ((y != one_class).sum())

                estimator.fit(X, y_class, sample_weight = weights)

            else:

                estimator.fit(X, y_class, sample_weight)

            self.estimators_[one_class] = estimator

    def predict_proba(self, X):
        """
        Predict probabilities to belong to a class for the each class.
        :param X: the estimator's format, data
        :return: ndarray, shape = [n_samples, n_classes], probabiities.
        """

        probas = numpy.zeros((len(X), len(self.classes)))

        for num, one_class in enumerate(self.classes):

            one_proba = self.estimators_[one_class].predict_proba(X)[:, 1]

            probas[:, num] = one_proba

        return probas

    def predict(self, X):
        """
        Predict classes.
        :param X: the estimator's format, data
        :return: array, shape = [n_samples], class labels [0, 1, 2, ..., n_classes - 1]
        """

        probas = self.predict_proba(X)

        predictions = probas.argmax(axis=1)

        return predictions


