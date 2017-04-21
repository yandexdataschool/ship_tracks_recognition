__author__ = 'Mikhail Hushchyn'

import ROOT
import numpy
import shipunit  as u

from recognition2d import track_pattern_recognition
from utils import Digitization, get_track_ids, get_fitted_trackids
from track_fit import track_fit


def execute(smeared_hits, stree, reco_mc_tracks, ShipGeo):

    X = Digitization(stree, smeared_hits)
    y = get_track_ids(stree, smeared_hits)

    ######################################## Do Track Pattern Recognition ##############################################

    reco_tracks = track_pattern_recognition(X, z_magnet=ShipGeo.Bfield.z, method='FastHough')

    ######################################### Fit recognized tracks ####################################################

    geoMat =  ROOT.genfit.TGeoMaterialInterface()
    bfield = ROOT.genfit.BellField(ShipGeo.Bfield.max ,ShipGeo.Bfield.z,2, ShipGeo.Yheight/2.*u.m)
    fM = ROOT.genfit.FieldManager.getInstance()
    fM.init(bfield)
    ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
    fitter = ROOT.genfit.DAF()

    theTracks = track_fit(ShipGeo, fitter, reco_tracks)

    ######################################### Estimate true track ids ##################################################

    fittedtrackids, fittedtrackfrac = get_fitted_trackids(y, reco_tracks)

    return reco_tracks, theTracks, fittedtrackids, fittedtrackfrac
