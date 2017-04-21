__author__ = 'Mikhail Hushchyn'

import ROOT
import numpy


def initialize(fGeo, ShipGeo):

    debug = 0
    #creates a dictionary with z coordinates of layers
    #and variables with station start/end coordinates
    #to be called once at the beginning of the eventloop
    i1=1 #1st layer
    i2=16 #last layer
    zlayer={} #dictionary with z coordinates of station1,2 layers
    zlayerv2={} #z-positions for stereo views
    z34layer={} #dictionary with z coordinates of station3,4 layers
    z34layerv2={} #z-positions for stereo views
    TStation1StartZ=0.
    TStation4EndZ=0.
    VetoStationZ=0.
    VetoStationEndZ=0.

    fgeo=fGeo
    #z-positions of Y-view tracking
    #4 stations, 4 views (Y,u,v,Y); each view has 2 planes and each plane has 2 layers

    for i in range(i1,i2+1):
        TStationz = ShipGeo.TrackStation1.z
        if (i>8) :
            TStationz = ShipGeo.TrackStation2.z
        # Y: vnb=0 or 3
        vnb=0.
        if (i>4): vnb=3.
        if (i>8): vnb=0.
        if (i>12): vnb=3.
        lnb = 0.
        if (i % 2 == 0) : lnb=1.
        pnb=0.
        if (i==3 or i==4 or i==7 or i==8 or i==11 or i==12 or i==15 or i==16) : pnb=1.

        #z positions of Y view of stations
        Zpos = TStationz+(vnb-3./2.)*ShipGeo.strawtubes.DeltazView+(float(pnb)-1./2.)*ShipGeo.strawtubes.DeltazPlane+(float(lnb)-1./2.)*ShipGeo.strawtubes.DeltazLayer
        zlayer[i]=[Zpos]

    #z-positions for stereo views

    for i in range(i1,i2+1):
        TStationz = ShipGeo.TrackStation1.z
        if (i>8) :
            TStationz = ShipGeo.TrackStation2.z
        #stereo views: vnb=1 or 2
        vnb=1.
        if (i>4): vnb=2.
        if (i>8): vnb=1.
        if (i>12): vnb=2.
        lnb = 0.
        if (i % 2 == 0) : lnb=1.
        pnb=0.
        if (i==3 or i==4 or i==7 or i==8 or i==11 or i==12 or i==15 or i==16) : pnb=1.

        #z positions of u,v view of stations
        Zpos_u = TStationz+(vnb-3./2.)*ShipGeo.strawtubes.DeltazView+(float(pnb)-1./2.)*ShipGeo.strawtubes.DeltazPlane+(float(lnb)-1./2.)*ShipGeo.strawtubes.DeltazLayer
        zlayerv2[i]=[Zpos_u]


    for i in range(i1,i2+1):
        TStationz = ShipGeo.TrackStation3.z
        if (i>8) :
            TStationz = ShipGeo.TrackStation4.z
        # Y: vnb=0 or 3
        vnb=0.
        if (i>4): vnb=3.
        if (i>8): vnb=0.
        if (i>12): vnb=3.
        lnb = 0.
        if (i % 2 == 0) : lnb=1.
        pnb=0.
        if (i==3 or i==4 or i==7 or i==8 or i==11 or i==12 or i==15 or i==16) : pnb=1.

        #z positions of x1 view of stations
        Zpos = TStationz+(vnb-3./2.)*ShipGeo.strawtubes.DeltazView+(float(pnb)-1./2.)*ShipGeo.strawtubes.DeltazPlane+(float(lnb)-1./2.)*ShipGeo.strawtubes.DeltazLayer
        z34layer[i]=[Zpos]


    for i in range(i1,i2+1):
        #zlayerv2[i]=[i*100.+50.]
        TStationz = ShipGeo.TrackStation3.z
        if (i>8) :
            TStationz = ShipGeo.TrackStation4.z
        #stereo views: vnb=1 or 2
        vnb=1.
        if (i>4): vnb=2.
        if (i>8): vnb=1.
        if (i>12): vnb=2.
        lnb = 0.
        if (i % 2 == 0) : lnb=1.
        pnb=0.
        if (i==3 or i==4 or i==7 or i==8 or i==11 or i==12 or i==15 or i==16) : pnb=1.

        #z positions of u,v view of stations
        Zpos_u = TStationz+(vnb-3./2.)*ShipGeo.strawtubes.DeltazView+(float(pnb)-1./2.)*ShipGeo.strawtubes.DeltazPlane+(float(lnb)-1./2.)*ShipGeo.strawtubes.DeltazLayer
        z34layerv2[i]=[Zpos_u]

    VetoStationZ = ShipGeo.vetoStation.z
    if debug==1: print "VetoStation midpoint z=",VetoStationZ
    VetoStationEndZ=VetoStationZ+(ShipGeo.strawtubes.DeltazView+ShipGeo.strawtubes.OuterStrawDiameter)/2
    for i in range(1,5):
        if i==1: TStationz = ShipGeo.TrackStation1.z
        if i==2: TStationz = ShipGeo.TrackStation2.z
        if i==3: TStationz = ShipGeo.TrackStation3.z
        if i==4: TStationz = ShipGeo.TrackStation4.z
        if debug==1:
            print "TrackStation",i," midpoint z=",TStationz
        for vnb in range(0,4):
            for pnb in range (0,2):
                for lnb in range (0,2):
                    Zpos = TStationz+(vnb-3./2.)*ShipGeo.strawtubes.DeltazView+(float(pnb)-1./2.)*ShipGeo.strawtubes.DeltazPlane+(float(lnb)-1./2.)*ShipGeo.strawtubes.DeltazLayer

    TStation1StartZ=zlayer[1][0]-ShipGeo.strawtubes.OuterStrawDiameter/2
    TStation4EndZ=z34layer[16][0]+ShipGeo.strawtubes.OuterStrawDiameter/2

    return zlayer,zlayerv2,z34layer,z34layerv2,TStation1StartZ,TStation4EndZ,VetoStationZ,VetoStationEndZ



def getReconstructibleTracks(iEvent,sTree,sGeo, reconstructiblerequired, threeprong, TStation1StartZ,TStation4EndZ,VetoStationZ,VetoStationEndZ):
    debug = 0

    PDG=ROOT.TDatabasePDG.Instance()

    #returns a list of reconstructible tracks for this event
    #call this routine once for each event before smearing
    MCTrackIDs=[]
    rc = sTree.GetEvent(iEvent)
    nMCTracks = sTree.MCTrack.GetEntriesFast()

    if debug==1: print "event nbr",iEvent,"has",nMCTracks,"tracks"
    #1. MCTrackIDs: list of tracks decaying after the last tstation and originating before the first
    for i in reversed(range(nMCTracks)):
        atrack = sTree.MCTrack.At(i)
        #for 3 prong decays check if its a nu
        if threeprong == 1:
            if PDG.GetParticle(atrack.GetPdgCode()):
                if PDG.GetParticle(atrack.GetPdgCode()).GetName()[:5]=="nu_mu":
                    if (atrack.GetStartZ() < TStation1StartZ and  atrack.GetStartZ() > VetoStationEndZ) and i not in MCTrackIDs:
                        MCTrackIDs.append(i)
                else:
                    if atrack.GetStartZ() > TStation4EndZ :
                        motherId=atrack.GetMotherId()
                        if motherId > -1 :
                            mothertrack=sTree.MCTrack.At(motherId)
                            mothertrackZ=mothertrack.GetStartZ()
                            #this mother track is a HNL decay
                            #track starts inside the decay volume? (after veto, before 1 st tstation)
                            if mothertrackZ < TStation1StartZ and mothertrackZ > VetoStationEndZ:
                                if motherId not in MCTrackIDs:
                                    MCTrackIDs.append(motherId)
        else:
            #track endpoint after tstations?
            if atrack.GetStartZ() > TStation4EndZ :
                motherId=atrack.GetMotherId()
                if motherId > -1 :
                    mothertrack=sTree.MCTrack.At(motherId)
                    mothertrackZ=mothertrack.GetStartZ()
                    #this mother track is a HNL decay
                    #track starts inside the decay volume? (after veto, before 1 st tstation)
                    if mothertrackZ < TStation1StartZ and mothertrackZ > VetoStationEndZ:
                        if motherId not in MCTrackIDs:
                            MCTrackIDs.append(motherId)
    if debug==1: print "Tracks with origin in decay volume",MCTrackIDs
    if len(MCTrackIDs)==0: return MCTrackIDs

    #2. hitsinTimeDet: list of tracks with hits in TimeDet
    nVetoHits = sTree.vetoPoint.GetEntriesFast()
    hitsinTimeDet=[]
    for i in range(nVetoHits):
        avetohit = sTree.vetoPoint.At(i)
        #hit in TimeDet?
        if sGeo.FindNode(avetohit.GetX(),avetohit.GetY(),avetohit.GetZ()).GetName() == 'TimeDet_1':
            if avetohit.GetTrackID() not in hitsinTimeDet:
                hitsinTimeDet.append(avetohit.GetTrackID())

    #3. Remove tracks from MCTrackIDs that are not in hitsinTimeDet
    itemstoremove=[]
    for item in MCTrackIDs:
        if threeprong==1:
            #don't remove the nu
            if PDG.GetParticle(sTree.MCTrack.At(item).GetPdgCode()).GetName()[:5]!="nu_mu" and item not in hitsinTimeDet:
                itemstoremove.append(item)
        else :
            if item not in hitsinTimeDet:
                itemstoremove.append(item)
    for item in itemstoremove:
        MCTrackIDs.remove(item)

    if debug==1: print "Tracks with hits in timedet",MCTrackIDs
    if len(MCTrackIDs)==0: return MCTrackIDs
    #4. Find straws that have multiple hits
    nHits = sTree.strawtubesPoint.GetEntriesFast()
    hitstraws={}
    duplicatestrawhit=[]
    if debug==1: print "Nbr of Rawhits=",nHits

    for i in range(nHits):
        ahit = sTree.strawtubesPoint[i]
        if (str(ahit.GetDetectorID())[:1]=="5") :
            if debug==1: print "Hit in straw Veto detector. Rejecting."
            continue
        strawname=str(ahit.GetDetectorID())

        if hitstraws.has_key(strawname):
            #straw was already hit
            if ahit.GetX()>hitstraws[strawname][1]:
                #this hit has higher x, discard it
                duplicatestrawhit.append(i)
            else:
                #del hitstraws[strawname]
                duplicatestrawhit.append(hitstraws[strawname][0])
                hitstraws[strawname]=[i,ahit.GetX()]
        else:
            hitstraws[strawname]=[i,ahit.GetX()]

    #5. Split hits up by station and outside stations
    hits1={}
    hits2={}
    hits3={}
    hits4={}
    trackoutsidestations=[]
    for i in range(nHits):
        if i in  duplicatestrawhit:
            if debug==1: print "Duplicate hit",i,"not reconstructible, rejecting."
            continue
        ahit = sTree.strawtubesPoint[i]
        #is hit inside acceptance? if not mark the track as bad
        if (((ahit.GetX()/245.)**2 + (ahit.GetY()/495.)**2) >= 1.):
            if ahit.GetTrackID() not in trackoutsidestations:
                trackoutsidestations.append(ahit.GetTrackID())
        if ahit.GetTrackID() not in MCTrackIDs:
            #hit on not reconstructible track
            if debug==1: print "Hit not on reconstructible track. Rejecting."
            continue
            #group hits per tracking station, key = trackid
        if str(ahit.GetDetectorID())[:1]=="1" :
            if hits1.has_key(ahit.GetTrackID()):
                hits1[ahit.GetTrackID()]=[hits1[ahit.GetTrackID()][0],i]
            else:
                hits1[ahit.GetTrackID()]=[i]
        if str(ahit.GetDetectorID())[:1]=="2" :
            if hits2.has_key(ahit.GetTrackID()):
                hits2[ahit.GetTrackID()]=[hits2[ahit.GetTrackID()][0],i]
            else:
                hits2[ahit.GetTrackID()]=[i]
        if str(ahit.GetDetectorID())[:1]=="3" :
            if hits3.has_key(ahit.GetTrackID()):
                hits3[ahit.GetTrackID()]=[hits3[ahit.GetTrackID()][0],i]
            else:
                hits3[ahit.GetTrackID()]=[i]
        if str(ahit.GetDetectorID())[:1]=="4" :
            if hits4.has_key(ahit.GetTrackID()):
                hits4[ahit.GetTrackID()]=[hits4[ahit.GetTrackID()][0],i]
            else:
                hits4[ahit.GetTrackID()]=[i]

                #6. Make list of tracks with hits in in station 1,2,3 & 4
    tracks_with_hits_in_all_stations=[]
    for key in hits1.keys():
        if (hits2.has_key(key) and hits3.has_key(key) ) and hits4.has_key(key):
            if key not in tracks_with_hits_in_all_stations and key not in trackoutsidestations:
                tracks_with_hits_in_all_stations.append(key)
    for key in hits2.keys():
        if (hits1.has_key(key) and hits3.has_key(key) ) and hits4.has_key(key):
            if key not in tracks_with_hits_in_all_stations and key not in trackoutsidestations:
                tracks_with_hits_in_all_stations.append(key)
    for key in hits3.keys():
        if ( hits2.has_key(key) and hits1.has_key(key) ) and hits4.has_key(key):
            if key not in tracks_with_hits_in_all_stations and key not in trackoutsidestations:
                tracks_with_hits_in_all_stations.append(key)
    for key in hits4.keys():
        if (hits2.has_key(key) and hits3.has_key(key)) and hits1.has_key(key):
            if key not in tracks_with_hits_in_all_stations and key not in trackoutsidestations:
                tracks_with_hits_in_all_stations.append(key)

                #7. Remove tracks from MCTrackIDs with hits outside acceptance or doesn't have hits in all stations
    itemstoremove=[]
    for item in MCTrackIDs:
        if threeprong==1:
            #don't remove the nu
            if PDG.GetParticle(sTree.MCTrack.At(item).GetPdgCode()).GetName()[:5]!="nu_mu" and item not in tracks_with_hits_in_all_stations:
                itemstoremove.append(item)
        else:
            if item not in tracks_with_hits_in_all_stations:
                itemstoremove.append(item)
    for item in itemstoremove:
        MCTrackIDs.remove(item)

    if debug==1:
        print "tracks_with_hits_in_all_stations",tracks_with_hits_in_all_stations
        print "Tracks with hits in all stations & inside acceptance ellipse",MCTrackIDs
    if len(MCTrackIDs)==0: return MCTrackIDs
    nbrechits=0
    for i in range(nHits):
        if i in  duplicatestrawhit:
            continue
        nbrechits+=1
        ahit = sTree.strawtubesPoint[i]
        if ahit.GetTrackID()>-1 and ahit.GetTrackID() in MCTrackIDs:
            atrack = sTree.MCTrack.At(ahit.GetTrackID())
            for j in range(ahit.GetTrackID()+1,nMCTracks) :
                childtrack = sTree.MCTrack.At(j)
                if childtrack.GetMotherId() == ahit.GetTrackID():
                    trackmomentum=atrack.GetP()
                    trackweight=atrack.GetWeight()
                    #rc=h['reconstructiblemomentum'].Fill(trackmomentum,trackweight)
                    motherId=atrack.GetMotherId()
                    if motherId==1 :
                        HNLmomentum=sTree.MCTrack.At(1).GetP()
                        #rc=h['HNLmomentumvsweight'].Fill(trackweight,HNLmomentum)
                        if j==nMCTracks :
                            trackmomentum=atrack.GetP()
                            trackweight=atrack.GetWeight()
                            #rc=h['reconstructiblemomentum'].Fill(trackmomentum,trackweight)
                            if atrack.GetMotherId()==1 :
                                HNLmomentum=sTree.MCTrack.At(1).GetP()
                                #rc=h['HNLmomentumvsweight'].Fill(trackweight,HNLmomentum)
    itemstoremove=[]
    for item in MCTrackIDs:
        atrack = sTree.MCTrack.At(item)
        motherId=atrack.GetMotherId()
        if motherId != 2: #!!!!
            itemstoremove.append(item)
    for item in itemstoremove:
        MCTrackIDs.remove(item)
        if debug==1: print "After removing the non HNL track, MCTrackIDs",MCTrackIDs
    if debug==1: print "Tracks with HNL mother",MCTrackIDs

    #8. check if the tracks are HNL children
    mufound=0
    pifound=0
    nu_mufound=0
    itemstoremove=[]
    if MCTrackIDs:
        for item in MCTrackIDs:
            try:
                if PDG.GetParticle(sTree.MCTrack.At(item).GetPdgCode()).GetName()[:2]=="mu"   : mufound+=1
                if PDG.GetParticle(sTree.MCTrack.At(item).GetPdgCode()).GetName()[:2]=="pi"   : pifound+=1
                if PDG.GetParticle(sTree.MCTrack.At(item).GetPdgCode()).GetName()[:5]=="nu_mu":
                    nu_mufound+=1
                    itemstoremove.append(item)
            except:
                if debug==1: print "Unknown particle with pdg code:",sTree.MCTrack.At(item).GetPdgCode()
        if reconstructiblerequired == 1 :
            if mufound!=1  and pifound!=1:
                if debug==1: print "No reconstructible pion or muon."
                MCTrackIDs=[]
        if reconstructiblerequired == 2 :
            if threeprong == 1 :
                if mufound!=2 or nu_mufound!=1 :
                    if debug==1: print "No reconstructible mu-mu-nu."
                    MCTrackIDs=[]
                else:
                    #remove the neutrino from MCTrackIDs for the rest
                    for item in itemstoremove:
                        MCTrackIDs.remove(item)
            else:
                if mufound!=1 or pifound!=1 :
                    if debug==1: print "No reconstructible pion and muon."
                    MCTrackIDs=[]
    if len(MCTrackIDs)>0:
        #rc=h['nbrhits'].Fill(nHits)
        #rc=h['nbrtracks'].Fill(nMCTracks)
        pass
    if debug==1: print "Tracks with required HNL decay particles",MCTrackIDs
    return MCTrackIDs


def smearHits(sTree, ShipGeo, modules, no_amb=None):

    random = ROOT.TRandom()
    ROOT.gRandom.SetSeed(13)

    # smear strawtube points
    SmearedHits = []
    key = -1

    for ahit in sTree.strawtubesPoint:

        key+=1
        detID = ahit.GetDetectorID()
        top = ROOT.TVector3()
        bot = ROOT.TVector3()

        modules["Strawtubes"].StrawEndPoints(detID,bot,top)

        #distance to wire, and smear it.
        dw  = ahit.dist2Wire()
        smear = dw
        if not no_amb:
            random = ROOT.TRandom()
            smear = abs(random.Gaus(dw,ShipGeo.strawtubes.sigma_spatial))

        SmearedHits.append( {'digiHit':key,'xtop':top.x(),'ytop':top.y(),'z':top.z(),'xbot':bot.x(),'ybot':bot.y(),'dist':smear} )


    return SmearedHits

def Digitization(sTree, SmearedHits):

    Hits = []

    for i in range(len(SmearedHits)):
        xtop=SmearedHits[i]['xtop']
        xbot=SmearedHits[i]['xbot']
        ytop=SmearedHits[i]['ytop']
        ybot=SmearedHits[i]['ybot']
        ztop=SmearedHits[i]['z']
        zbot=SmearedHits[i]['z']
        distance=SmearedHits[i]['dist']
        detid=sTree.strawtubesPoint[i].GetDetectorID()

        ahit=[xtop, ytop, ztop, xbot, ybot, zbot, float(distance), int(detid)]
        Hits.append(ahit)

    return numpy.array(Hits)

def fracMCsame(trackids):

    track={}
    nh=len(trackids)
    for tid in trackids:
        if tid==999:
            nh-=1
            continue
        if track.has_key(tid):
            track[tid]+=1
        else:
            track[tid]=1
    #now get track with largest number of hits
    tmax=max(track, key=track.get)

    frac=0.
    if nh>0: frac=float(track[tmax])/float(nh)
    return frac,tmax

def get_track_ids(stree, smeared_hits):

    y = []

    for i in range(len(smeared_hits)):

        track_id = stree.strawtubesPoint[i].GetTrackID()
        y.append(track_id)

    return numpy.array(y)


def get_fitted_trackids(y, reco_tracks):

    fittedtrackids = []
    fittedtrackfrac = []

    for track_id in reco_tracks.keys():

        frac, tmax = fracMCsame(y[reco_tracks[track_id]['hits']])
        fittedtrackids.append(tmax)
        fittedtrackfrac.append(frac)

    return fittedtrackids, fittedtrackfrac