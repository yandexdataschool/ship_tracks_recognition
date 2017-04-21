__author__ = 'Mikhail Hushchyn'

import ROOT
import numpy

import getopt
import sys

# For ShipGeo
from ShipGeoConfig import ConfigRegistry
from rootpyPickler import Unpickler

# For modules
import shipDet_conf

# For track pattern recognition
from utils import initialize, getReconstructibleTracks, smearHits
from execute import execute

# For track pattern recognition quality measure
from quality import init_book_hist, quality_metrics, save_hists



def run_track_pattern_recognition(input_file, geo_file, dy, model='FastHough'):


    ############################################# Load SHiP geometry ###################################################

    # Check geo file
    try:
        fgeo = ROOT.TFile(geo_file)
    except:
        print "An error with opening the ship geo file."
        raise

    sGeo = fgeo.FAIRGeom

    # Prepare ShipGeo dictionary
    if not fgeo.FindKey('ShipGeo'):

        if sGeo.GetVolume('EcalModule3') :
            ecalGeoFile = "ecal_ellipse6x12m2.geo"
        else:
            ecalGeoFile = "ecal_ellipse5x10m2.geo"

        if dy:
            ShipGeo = ConfigRegistry.loadpy("$FAIRSHIP/geometry/geometry_config.py", Yheight = dy, EcalGeoFile = ecalGeoFile)
        else:
            ShipGeo = ConfigRegistry.loadpy("$FAIRSHIP/geometry/geometry_config.py", EcalGeoFile = ecalGeoFile)

    else:
        upkl    = Unpickler(fgeo)
        ShipGeo = upkl.load('ShipGeo')

    ############################################# Load SHiP modules ####################################################

    run = ROOT.FairRunSim()
    modules = shipDet_conf.configure(run,ShipGeo)

    ############################################# Load inpur data file #################################################

    # Check input file
    try:
        fn = ROOT.TFile(input_file,'update')
    except:
        print "An error with opening the input data file."
        raise

    sTree = fn.cbmsim

    ############################## Initialize SHiP Spectrometer Tracker geometry #######################################

    zlayer, \
    zlayerv2, \
    z34layer, \
    z34layerv2, \
    TStation1StartZ, \
    TStation4EndZ, \
    VetoStationZ, \
    VetoStationEndZ = initialize(fgeo, ShipGeo)


    ########################################## Start Track Pattern Recognition #########################################

    # Init book of hists for the quality measurements
    h = init_book_hist()

    # Start event loop
    nEvents   = sTree.GetEntries()
    for iEvent in range(nEvents):

        if iEvent%10 == 0:
            print 'Event ', iEvent

        # Take one event
        rc = sTree.GetEvent(iEvent)

        # Find reconstructible tracks in the event
        reco_mc_tracks = getReconstructibleTracks(iEvent, sTree, sGeo, 2, 0,
                                                  TStation1StartZ, TStation4EndZ, VetoStationZ, VetoStationEndZ) # TODO:!!!

        # Smear hits of the event. Only for MC data.
        smeared_hits = smearHits(sTree, modules, no_amb=None)

        # Do Track Pattern Recognition
        reco_tracks, \
        theTracks, \
        fittedtrackids, \
        fittedtrackfrac = execute(smeared_hits, sTree, reco_mc_tracks, ShipGeo)

        # Measure Track Pattern Recognition Quality
        quality_metrics(smeared_hits, sTree, reco_mc_tracks, reco_tracks, h)


    # Save results
    save_hists(h, 'hists.root')


    return

if __name__ == "__main__":

    input_file = None
    geo_file = None
    dy = None
    model = None


    argv = sys.argv[1:]

    msg = '''Predicts PID MVA outputs for a given input file.\n\
    Usage:\n\
      python run.py [options] \n\
      -h  --help                    : Shows this help
      '''

    try:
        opts, args = getopt.getopt(argv, "hm:i:g:",
                                   ["help", "model=", "input=", "geo="])
    except getopt.GetoptError:
        print "Wrong options were used. Please, read the following help:\n"
        print msg
        sys.exit(2)
    if len(argv) == 0:
        print msg
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print msg
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-g", "--geo"):
            geo_file = arg


    run_track_pattern_recognition(input_file, geo_file, dy, model='FastHough')


