__author__ = 'mikhail91'

import numpy
import pandas


class Combinator(object):

    def __init__(self, z_magnet=3070., magnetic_field=-0.75, dy_max=2., dx_max=20.):
        """
        This class combines tracks before and after the magnet,
        estimates a particles charge and momentim based on its deflection in the magnetic field.
        :param z_magnet: floaf, z-coordinate of the center of the magnet.
        :param magnetic_field: float, inductivity of the magnetic field.
        :param dy_max: float, max distance on y between the tracks before and after the magnet in center of the magnet.
        :param dx_max: float, max distance on x between the tracks before and after the magnet in center of the magnet.
        :return:
        """

        self.z_magnet = z_magnet
        self.magnetic_field = magnetic_field
        self.dy_max = dy_max
        self.dx_max = dx_max

        self.tracks_combinations_ = None
        self.charges_ = None
        self.inv_momentums_ = None

    def get_tracks_combination(self, tracks_before, tracks_after):
        """
        This function combines the two tracks.
        :param tracks_before: list of [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track before the magnet. y = kx + b.
        :param tracks_after: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track after the magnet. y = kx + b.
        :return: list of [track_id_before, track_id_after]
        """

        z_magnet = self.z_magnet
        dy_max = self.dy_max
        dx_max = self.dx_max

        self.dx = []
        self.dy = []

        tracks_combinations = []

        for track_id_before, one_track_before in enumerate(tracks_before):

            for track_id_after, one_track_after in enumerate(tracks_after):


                if len(one_track_before[1])==0 or len(one_track_after[1])==0:
                    continue

                y_before = z_magnet * one_track_before[0][0] + one_track_before[0][1]
                x_before = z_magnet * one_track_before[1][0] + one_track_before[1][1]

                y_after = z_magnet * one_track_after[0][0] + one_track_after[0][1]
                x_after = z_magnet * one_track_after[1][0] + one_track_after[1][1]

                dy = numpy.abs(y_after - y_before)
                dx = numpy.abs(x_after - x_before)
                dr = numpy.sqrt(dy**2 + dx**2)

                #print dy, dx


                if dy <= dy_max and dx <= dx_max:

                    self.dx.append(x_after - x_before)
                    self.dy.append(y_after - y_before)

                    tracks_combinations.append(numpy.array([track_id_before, track_id_after]))
                    #continue

        return numpy.array(tracks_combinations)

    def get_charges(self, tracks_before, tracks_after, tracks_combinations):
        """
        This function estimates the charges of the particles.
        :param tracks_before: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track before the magnet. y = kx + b.
        :param tracks_after: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track after the magnet. y = kx + b.
        :param tracks_combinations: list of [track_id_before, track_id_after], indexes of the tracks.
        :return: list if estimated charges
        """

        charges = []

        for one_tracks_combination in tracks_combinations:

            one_track_before = tracks_before[one_tracks_combination[0]]
            one_track_after = tracks_after[one_tracks_combination[1]]

            k_yz_before = one_track_before[0][0]
            k_yz_after = one_track_after[0][0]

            difftan = (k_yz_before - k_yz_after) / (1. + k_yz_before * k_yz_after)


            if difftan > 0:

                one_charge = -1.

            else:

                one_charge = 1.


            charges.append(one_charge)

        return numpy.array(charges)


    def get_inv_momentums(self, tracks_before, tracks_after, tracks_combinations):
        """
        This function estimates the momentums of the particles.
        :param tracks_before: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track before the magnet. y = kx + b.
        :param tracks_after: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track after the magnet. y = kx + b.
        :param tracks_combinations: list of [track_id_before, track_id_after], indexes of the tracks.
        :return: list if estimated inverse momentums.
        """

        Bm = self.magnetic_field
        inv_momentums = []

        for one_tracks_combination in tracks_combinations:

            one_track_before = tracks_before[one_tracks_combination[0]]
            one_track_after = tracks_after[one_tracks_combination[1]]

            k_yz_before = one_track_before[0][0]
            k_yz_after = one_track_after[0][0]

            a = numpy.arctan(k_yz_before)
            b = numpy.arctan(k_yz_after)
            pinv = numpy.sin(a - b) / (0.3 * Bm)

            #pinv = numpy.abs(pinv) # !!!!

            inv_momentums.append(pinv)


        return numpy.array(inv_momentums)


    def combine(self, tracks_before, tracks_after):
        """
        Run the combinator.
        :param tracks_before: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track before the magnet. y = kx + b.
        :param tracks_after: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track after the magnet. y = kx + b.
        :return:
        """

        tracks_combinations = self.get_tracks_combination(tracks_before, tracks_after)
        charges = self.get_charges(tracks_before, tracks_after, tracks_combinations)
        inv_momentums = self.get_inv_momentums(tracks_before, tracks_after, tracks_combinations)


        self.tracks_combinations_ = tracks_combinations
        self.charges_ = charges
        self.inv_momentums_ = inv_momentums


class SuperCombinator(Combinator):

    """
        This class combines tracks before and after the magnet using a classifier,
        estimates a particles charge and momentim based on its deflection in the magnetic field.
        :params classifier: scikit-learn-like classifier, which predicts that
        two tracks candidates before and after the magnet belongs the same track
        :param z_magnet: floaf, z-coordinate of the center of the magnet.
        :param magnetic_field: float, inductivity of the magnetic field.
        :param dy_max: float, max distance on y between the tracks before and after the magnet in center of the magnet.
        :param dx_max: float, max distance on x between the tracks before and after the magnet in center of the magnet.
        :return:
        """

    def __init__(self, classifier=None, z_magnet=3070., magnetic_field=-0.75):

        Combinator.__init__(self, z_magnet, magnetic_field)

        self.classifier = classifier

    def data_collection(self, tracks, events):
        """
        The method returns data for the classifier training.
        :param tracks: dict, {event_id: 'labels12':[-1, 1, 0, -1, ...], 'params12':[[[k_y, b_y], [k_stereo, b_stereo]], ...],
                           'labels34':[-1, 1, 0, -1, ...], 'params34':[[[k_y, b_y], [k_stereo, b_stereo]], ...]}
        :param events: pandas.DataFrame(), True MC of the hits.
        :return: pandas.DataFrame()
        """

        combination_data = pandas.DataFrame(columns=[u'dx', u'dy', 'dr',
                                                      u'k_xz_12', u'b_xz_12',
                                                      u'k_yz_12',u'b_yz_12',
                                                      u'k_xz_34', u'b_xz_34',
                                                      u'k_yz_34', u'b_yz_34',
                                                      'dk_xz', 'db_xz',
                                                      'dk_yz', 'db_yz', 'label'])

        for event_id in tracks.keys():

            event = events[events.EventID == event_id]
            track = tracks[event_id]

            event12 = event[(event.StatNb == 1) + (event.StatNb == 2)]
            event34 = event[(event.StatNb == 3) + (event.StatNb == 4)]


            true_labels12 = event12.Label.values
            true_labels34 = event34.Label.values

            labels12 = track['labels12']
            labels34 = track['labels34']
            params12 = track['params12']
            params34 = track['params34']

            for before in range(len(params12)):

                for after in range(len(params34)):

                    if len(params12[before][1]) == 0 or len(params34[after][1]) == 0:
                        continue


                    k_xz_12, b_xz_12 = params12[before][1][0], params12[before][1][1]
                    k_yz_12, b_yz_12 = params12[before][0][0], params12[before][0][1]
                    k_xz_34, b_xz_34 = params34[after][1][0], params34[after][1][1]
                    k_yz_34, b_yz_34 = params34[after][0][0], params34[after][0][1]

                    dk_xz = numpy.abs(params34[after][1][0] - params12[before][1][0])
                    db_xz = numpy.abs(params34[after][1][1] - params12[before][1][1])

                    dk_yz = numpy.abs(params34[after][0][0] - params12[before][0][0])
                    db_yz = numpy.abs(params34[after][0][1] - params12[before][0][1])

                    y_before = self.z_magnet * k_yz_12 + b_yz_12
                    x_before = self.z_magnet * k_xz_12 + b_xz_12

                    y_after = self.z_magnet * k_yz_34 + b_yz_34
                    x_after = self.z_magnet * k_xz_34 + b_xz_34

                    dy = numpy.abs(y_after - y_before)
                    dx = numpy.abs(x_after - x_before)
                    dr = numpy.sqrt(dx**2 + dy**2)


                    unique_before, counts_before = numpy.unique(true_labels12[labels12 == before],
                                                                return_counts=True)
                    max_fraction_true_label_before = unique_before[counts_before == counts_before.max()][0]

                    unique_after, counts_after = numpy.unique(true_labels34[labels34 == after],
                                                              return_counts=True)
                    max_fraction_true_label_after = unique_after[counts_after == counts_after.max()][0]

                    if max_fraction_true_label_before == max_fraction_true_label_after:
                        label = 1
                    else:
                        label = 0

                    combination_data.loc[len(combination_data)] = [dx, dy, dr,
                                                                   k_xz_12, b_xz_12,
                                                                   k_yz_12, b_yz_12,
                                                                   k_xz_34, b_xz_34,
                                                                   k_yz_34, b_yz_34,
                                                                   dk_xz, db_xz,
                                                                   dk_yz, db_yz, label]

        return combination_data


    def get_tracks_combination(self, tracks_before, tracks_after):
        """
        This function combines the two tracks.
        :param tracks_before: list of [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track before the magnet. y = kx + b.
        :param tracks_after: list, [[k_yz, b_yz], [k_xz, b_xz]], parameters of the track after the magnet. y = kx + b.
        :return: list of [track_id_before, track_id_after]
        """

        z_magnet = self.z_magnet
        dy_max = self.dy_max
        dx_max = self.dx_max

        tracks_combinations = []

        for track_id_before, one_track_before in enumerate(tracks_before):

            for track_id_after, one_track_after in enumerate(tracks_after):


                if len(one_track_before[1])==0 or len(one_track_after[1])==0:
                    continue


                k_xz_12, b_xz_12 = one_track_before[1][0], one_track_before[1][1]
                k_yz_12, b_yz_12 = one_track_before[0][0], one_track_before[0][1]
                k_xz_34, b_xz_34 = one_track_after[1][0], one_track_after[1][1]
                k_yz_34, b_yz_34 = one_track_after[0][0], one_track_after[0][1]

                dk_xz = numpy.abs(k_xz_34 - k_xz_12)
                db_xz = numpy.abs(b_xz_34 - b_xz_12)

                dk_yz = numpy.abs(k_yz_34 - k_yz_12)
                db_yz = numpy.abs(b_yz_34 - b_yz_12)

                y_before = self.z_magnet * k_yz_12 + b_yz_12
                x_before = self.z_magnet * k_xz_12 + b_xz_12

                y_after = self.z_magnet * k_yz_34 + b_yz_34
                x_after = self.z_magnet * k_xz_34 + b_xz_34

                y_before = z_magnet * one_track_before[0][0] + one_track_before[0][1]
                x_before = z_magnet * one_track_before[1][0] + one_track_before[1][1]

                y_after = z_magnet * one_track_after[0][0] + one_track_after[0][1]
                x_after = z_magnet * one_track_after[1][0] + one_track_after[1][1]

                dy = numpy.abs(y_after - y_before)
                dx = numpy.abs(x_after - x_before)
                dr = numpy.sqrt(dy**2 + dx**2)

                #print dy, dx

                x_clf = numpy.array([[dx, dy, dr,
                                   k_xz_12, b_xz_12,
                                   k_yz_12, b_yz_12,
                                   k_xz_34, b_xz_34,
                                   k_yz_34, b_yz_34,
                                   dk_xz, db_xz,
                                   dk_yz, db_yz]])

                label = self.classifier.predict(x_clf)


                if label[0] == 1:

                    tracks_combinations.append(numpy.array([track_id_before, track_id_after]))
                    continue

        return numpy.array(tracks_combinations)




from numpy.linalg import inv

class GlobalFit(object):

    def __init__(self, z_magnet=3070):

        self.z_magnet = z_magnet

    def fit(self, x1, y1, x2, y2, w1=None, w2=None):

        xm = self.z_magnet

        xx1 = numpy.concatenate((x1.reshape(-1, 1) - xm,
                                 numpy.zeros((len(x1), 1)),
                                 numpy.ones((len(x1), 1))), axis=1)
        xx2 = numpy.concatenate((x2.reshape(-1, 1) - xm,
                                 x2.reshape(-1, 1) - xm,
                                 numpy.ones((len(x2), 1))), axis=1)

        yy1 = y1.reshape(-1, 1)
        yy2 = y2.reshape(-1, 1)


        X = numpy.matrix(numpy.concatenate((xx1, xx2), axis=0))
        Y = numpy.matrix(numpy.concatenate((yy1, yy2), axis=0))

        if w1==None or w2==None:
            W = numpy.matrix(numpy.eye(len(x1) + len(x2)))
        else:
            ww1 = w1.reshape(-1, 1)
            ww2 = w2.reshape(-1, 1)
            ww = numpy.concatenate((ww1, ww2), axis=0)
            W = numpy.matrix(numpy.eye(len(ww)) * ww)

        a = inv(X.T * W * X) * X.T * W * Y

        return numpy.array(a).reshape(-1)


class MomentumCorrecter(object):

    def __init__(self, z_magnet=3070., magnetic_field=-0.75):

        self.z_magnet = z_magnet
        self.magnetic_field = magnetic_field

    def fit(self, labels_before, labels_after, tracks_combinations, event_before, event_after, weights_before=None, weights_after=None):

        Bm = self.magnetic_field
        inv_momentums = []

        for one_tracks_combination in tracks_combinations:

            track_before = event_before[(event_before.IsStereo.values==0)*(labels_before == one_tracks_combination[0])]
            track_after = event_after[(event_after.IsStereo.values==0)*(labels_after == one_tracks_combination[1])]

            x1 = track_before.Wz1.values
            y1 = track_before.Wy1.values
            x2 = track_after.Wz1.values
            y2 = track_after.Wy1.values

            if weights_before==None or weights_after==None:
                w1 = None
                w2 = None
            else:
                w1 = weights_before[(event_before.IsStereo.values==0)*(labels_before == one_tracks_combination[0])]
                w2 = weights_after[(event_after.IsStereo.values==0)*(labels_after == one_tracks_combination[1])]

            gf = GlobalFit()
            k, dk, ym = gf.fit(x1, y1, x2, y2, w1, w2)

            pinv = numpy.sin(-dk) / (0.3 * Bm)

            #pinv = numpy.abs(pinv) # !!!!

            inv_momentums.append(pinv)

        self.inv_momentums_ = numpy.array(inv_momentums)


class SimplifiedCombinator(object):

    def __init__(self, z_magnet=3070., magnetic_field=-0.75, dy_max=2.):
        """
        This class combines tracks before and after the magnet,
        estimates a particles charge and momentim based on its deflection in the magnetic field.
        :param z_magnet: floaf, z-coordinate of the center of the magnet.
        :param magnetic_field: float, inductivity of the magnetic field.
        :param dy_max: float, max distance on y between the tracks before and after the magnet in center of the magnet.
        :param dx_max: float, max distance on x between the tracks before and after the magnet in center of the magnet.
        :return:
        """

        self.z_magnet = z_magnet
        self.magnetic_field = magnetic_field
        self.dy_max = dy_max

    def get_tracks_combination(self, tracks_before, tracks_after):
        """
        This function combines the two tracks.
        :param tracks_before: list of [k_yz, b_yz], parameters of the track before the magnet. y = kx + b.
        :param tracks_after: list, [k_yz, b_yz], parameters of the track after the magnet. y = kx + b.
        :return: list of [track_id_before, track_id_after]
        """

        z_magnet = self.z_magnet
        dy_max = self.dy_max

        self.dy = []

        tracks_combinations = []

        for track_id_before, one_track_before in enumerate(tracks_before):

            for track_id_after, one_track_after in enumerate(tracks_after):


                if len(one_track_before)==0 or len(one_track_after)==0:
                    continue

                y_before = z_magnet * one_track_before[0] + one_track_before[1]
                y_after = z_magnet * one_track_after[0] + one_track_after[1]

                dy = numpy.abs(y_after - y_before)

                if dy <= dy_max:

                    self.dy.append(y_after - y_before)

                    tracks_combinations.append(numpy.array([track_id_before, track_id_after]))
                    #continue

        return numpy.array(tracks_combinations)