__author__ = 'Alenkin Oleg, Mikhail Hushchyn'

import numpy as np
import pandas as pd
from math import *


class ParametresYZ:

    def __init__(self, y, dist2Wire, index, used):
        self.dist2Wire = dist2Wire
        self.y = y
        self.index = index
        self.used = used


def selector(data, StatNb, ViewNb, PlaneNb, LayerNb):
    return data.loc[(data.StatNb==StatNb)&(data.ViewNb==ViewNb)&(data.PlaneNb==PlaneNb)&(data.LayerNb==LayerNb)]


def get_plane(point1, point2):
    """
    Finds parametres of the line y = k * x + b.
    
    Args:
        point1: any point of the line represented as [x, y]
        point2: another point of the line
    
    Returns:
        tuple k, b; where k - tan(alpha), b - bias
    """

    y1 = point1[0]
    z1 = point1[1]
    y2 = point2[0]
    z2 = point2[1]
    
    k = float(y2 - y1)/float(z2 - z1)
    b = y1 - k*z1

    return k, b

def modify_for_yz_analysis_1_2(event):
    """
    Gets table of hits, fetchs only hits from Y-views (1 & 2 stations) and adds to it columns 'Wz', 'Wy' - coordinates 
    of centres of tubes in plane (z, y).
    
    Args:
        event: pd.DataFrame() that necesserily contains this columns: 
        'StatNb' - number of station,
        'ViewNb' - number of view,
        'PlaneNb' - number of plane,
        'LayerNb' - number of layer,
        'StrawNb' - number of straw.
        
    Returns:
        the same pd.DataFrame() but with 2 new columns: 'Wz', 'Wy'
    """
    layer1000 = selector(event, 1, 0, 0, 0)
    layer1001 = selector(event, 1, 0, 0, 1)
    layer1010 = selector(event, 1, 0, 1, 0)
    layer1011 = selector(event, 1, 0, 1, 1)

    layer1300 = selector(event, 1, 3, 0, 0)
    layer1301 = selector(event, 1, 3, 0, 1)
    layer1310 = selector(event, 1, 3, 1, 0)
    layer1311 = selector(event, 1, 3, 1, 1)

    layer2000 = selector(event, 2, 0, 0, 0)
    layer2001 = selector(event, 2, 0, 0, 1)
    layer2010 = selector(event, 2, 0, 1, 0)
    layer2011 = selector(event, 2, 0, 1, 1)

    layer2300 = selector(event, 2, 3, 0, 0)
    layer2301 = selector(event, 2, 3, 0, 1)
    layer2310 = selector(event, 2, 3, 1, 0)
    layer2311 = selector(event, 2, 3, 1, 1)
    

    #1-y1
    z1000 = 2598. - 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1000.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer1000.StrawNb.values-1)) + 499 + 0.44
    layer1000.loc[:, 'Wz'] = np.ones(layer1000.shape[0])*z1000

    z1001 = z1000 + 1.1
    layer1001.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer1001.StrawNb.values-1)) + 499 + 0.44
    layer1001.loc[:, 'Wz'] = np.ones(layer1001.shape[0])*z1001

    z1010 = z1000 + 2.6
    layer1010.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer1010.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1010.loc[:, 'Wz'] = np.ones(layer1010.shape[0])*z1010

    z1011 = z1010 + 1.1
    layer1011.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1011.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1011.loc[:, 'Wz'] = np.ones(layer1011.shape[0])*z1011

    #1-y2
    z1300 = 2598. + 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1300.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer1300.StrawNb.values-1)) + 499 + 0.44
    layer1300.loc[:, 'Wz'] = np.ones(layer1300.shape[0])*z1300

    z1301 = z1300 + 1.1
    layer1301.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer1301.StrawNb.values-1)) + 499 + 0.44
    layer1301.loc[:, 'Wz'] = np.ones(layer1301.shape[0])*z1301

    z1310 = z1300 + 2.6
    layer1310.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer1310.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1310.loc[:, 'Wz'] = np.ones(layer1310.shape[0])*z1310

    z1311 = z1310 + 1.1
    layer1311.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1311.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1311.loc[:, 'Wz'] = np.ones(layer1311.shape[0])*z1311

    #2-y1
    z2000 = 2798. - 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2000.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer2000.StrawNb.values-1)) + 499 + 0.44
    layer2000.loc[:, 'Wz'] = np.ones(layer2000.shape[0])*z2000

    z2001 = z2000 + 1.1
    layer2001.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer2001.StrawNb.values-1)) + 499 + 0.44
    layer2001.loc[:, 'Wz'] = np.ones(layer2001.shape[0])*z2001

    z2010 = z2000 + 2.6
    layer2010.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer2010.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2010.loc[:, 'Wz'] = np.ones(layer2010.shape[0])*z2010

    z2011 = z2010 + 1.1
    layer2011.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2011.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2011.loc[:, 'Wz'] = np.ones(layer2011.shape[0])*z2011

    #2-y2
    z2300 = 2798. + 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2300.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer2300.StrawNb.values-1)) + 499 + 0.44
    layer2300.loc[:, 'Wz'] = np.ones(layer2300.shape[0])*z2300

    z2301 = z2300 + 1.1
    layer2301.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer2301.StrawNb.values-1)) + 499 + 0.44
    layer2301.loc[:, 'Wz'] = np.ones(layer2301.shape[0])*z2301

    z2310 = z2300 + 2.6
    layer2310.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer2310.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2310.loc[:, 'Wz'] = np.ones(layer2310.shape[0])*z2310

    z2311 = z2310 + 1.1
    layer2311.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2311.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2311.loc[:, 'Wz'] = np.ones(layer2311.shape[0])*z2311
    
    layers = [layer1000, layer1001, layer1010, layer1011,
              layer1300, layer1301, layer1310, layer1311,
              layer2000, layer2001, layer2010, layer2011,
              layer2300, layer2301, layer2310, layer2311]

    return pd.concat(layers, axis=0)



def modify_for_yz_analysis_3_4(event):
    """
    Gets table of hits, fetchs only hits from Y-views (3 & 4 stations) and adds to it columns 'Wz', 'Wy' - coordinates 
    of centres of tubes in plane (z, y).
    
    Args:
        event: pd.DataFrame() that necesserily contains this columns: 
        'StatNb' - number of station,
        'ViewNb' - number of view,
        'PlaneNb' - number of plane,
        'LayerNb' - number of layer,
        'StrawNb' - number of straw.
        
    Returns:
        the same pd.DataFrame() but with 2 new columns: 'Wz', 'Wy'
    """
    layer3000 = selector(event, 3, 0, 0, 0)
    layer3001 = selector(event, 3, 0, 0, 1)
    layer3010 = selector(event, 3, 0, 1, 0)
    layer3011 = selector(event, 3, 0, 1, 1)

    layer3300 = selector(event, 3, 3, 0, 0)
    layer3301 = selector(event, 3, 3, 0, 1)
    layer3310 = selector(event, 3, 3, 1, 0)
    layer3311 = selector(event, 3, 3, 1, 1)

    layer4000 = selector(event, 4, 0, 0, 0)
    layer4001 = selector(event, 4, 0, 0, 1)
    layer4010 = selector(event, 4, 0, 1, 0)
    layer4011 = selector(event, 4, 0, 1, 1)

    layer4300 = selector(event, 4, 3, 0, 0)
    layer4301 = selector(event, 4, 3, 0, 1)
    layer4310 = selector(event, 4, 3, 1, 0)
    layer4311 = selector(event, 4, 3, 1, 1)
    
    #3-y1
    z3000 = 3338. - 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer3000.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer3000.StrawNb.values-1)) + 499 + 0.44
    layer3000.loc[:, 'Wz'] = np.ones(layer3000.shape[0])*z3000

    z3001 = z3000 + 1.1
    layer3001.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer3001.StrawNb.values-1)) + 499 + 0.44
    layer3001.loc[:, 'Wz'] = np.ones(layer3001.shape[0])*z3001

    z3010 = z3000 + 2.6
    layer3010.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer3010.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3010.loc[:, 'Wz'] = np.ones(layer3010.shape[0])*z3010

    z3011 = z3010 + 1.1
    layer3011.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer3011.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3011.loc[:, 'Wz'] = np.ones(layer3011.shape[0])*z3011

    #3-y2
    z3300 = 3338. + 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer3300.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer3300.StrawNb.values-1)) + 499 + 0.44
    layer3300.loc[:, 'Wz'] = np.ones(layer3300.shape[0])*z3300

    z3301 = z3300 + 1.1
    layer3301.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer3301.StrawNb.values-1)) + 499 + 0.44
    layer3301.loc[:, 'Wz'] = np.ones(layer3301.shape[0])*z3301

    z3310 = z3300 + 2.6
    layer3310.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer3310.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3310.loc[:, 'Wz'] = np.ones(layer3310.shape[0])*z3310

    z3311 = z3310 + 1.1
    layer3311.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer3311.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3311.loc[:, 'Wz'] = np.ones(layer3311.shape[0])*z3311

    #4-y1
    z4000 = 3538. - 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer4000.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer4000.StrawNb.values-1)) + 499 + 0.44
    layer4000.loc[:, 'Wz'] = np.ones(layer4000.shape[0])*z4000

    z4001 = z4000 + 1.1
    layer4001.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer4001.StrawNb.values-1)) + 499 + 0.44
    layer4001.loc[:, 'Wz'] = np.ones(layer4001.shape[0])*z4001

    z4010 = z4000 + 2.6
    layer4010.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer4010.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4010.loc[:, 'Wz'] = np.ones(layer4010.shape[0])*z4010

    z4011 = z4010 + 1.1
    layer4011.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer4011.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4011.loc[:, 'Wz'] = np.ones(layer4011.shape[0])*z4011

    #4-y2
    z4300 = 3538. + 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer4300.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer4300.StrawNb.values-1)) + 499 + 0.44
    layer4300.loc[:, 'Wz'] = np.ones(layer4300.shape[0])*z4300

    z4301 = z4300 + 1.1
    layer4301.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer4301.StrawNb.values-1)) + 499 + 0.44
    layer4301.loc[:, 'Wz'] = np.ones(layer4301.shape[0])*z4301

    z4310 = z4300 + 2.6
    layer4310.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer4310.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4310.loc[:, 'Wz'] = np.ones(layer4310.shape[0])*z4310

    z4311 = z4310 + 1.1
    layer4311.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer4311.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4311.loc[:, 'Wz'] = np.ones(layer4311.shape[0])*z4311
    
    layers = [layer3000, layer3001, layer3010, layer3011,
              layer3300, layer3301, layer3310, layer3311,
              layer4000, layer4001, layer4010, layer4011,
              layer4300, layer4301, layer4310, layer4311]

    return pd.concat(layers, axis=0)



def conventor_yz(event, indicator):
    """
    Gets pd.DataFrame() and transforms it into dictionary.
    
    Args:
        event: pd.DataFrame() that necesserily contains columns 'Wz', 'Wy', 'dist2Wire';
        indicator: 0 = 1 & 2 stations, 1 = 3 & 4 stations.
    Returns:
        dictionary: keys are values of 'Wz'; values are stuctures with fields(y, dist2Wire, index, used).
    """

    if (indicator):

        event = modify_for_yz_analysis_3_4(event)

    else:

        event = modify_for_yz_analysis_1_2(event)



    dictionary = {}

    for i in event.index:

        params = ParametresYZ(event.Wy[i], event.dist2Wire[i], event.Index[i], False)
        dictionary.setdefault(event.Wz[i], []).append(params)


    return dictionary



def dist2Track(point, track):
    return np.abs(- track[0] * point[0] + point[1] - track[1]) / np.sqrt(track[0]**2 + 1.0)



def points_crossing_line_yz_new(plane_k, plane_b, plane_width, hits, n_min, clf, event):
    """
    Counts the number of points which intercept line with parametres: plane_k, plane_b, plane_width.
    If the result more than n_min than makes linnear regression on this points.
    
    Args:
        plane_k, plane_b, plane_width: parametres of line in 2d space (y, z). It's a hyperplane in 3d space;
        hits: dictionary that contains all hits, key is z coordinate of centre of tube, value is structure 
            with fields- y, dist2Wire, index, used;
        n_min: minimal number of hits intercepting a track.
    Returns:
        indicator, crossing_points, lin_regr;
        indicator: false- means line with this parametres doesn't cover a track,
            true- line covers a track;
        crossing_points: array of indexes of points that determine the track;
        lin_regr: parametres k, b of linnear regression on crossing points.
    """

    marks = {}
    crossing_points = []
    Y = [] # for linear regression
    Z = []
    weights = []
    n = 0 # number of touched layers


    for z in hits:

        marks[z] = []
        indicator = False

        lower_y = plane_k * z + plane_b - 1. * plane_width #/ np.cos(np.arctan(plane_k))
        upper_y = plane_k * z + plane_b + 1. * plane_width #/ np.cos(np.arctan(plane_k))

        for j in range(len(hits[z])):

            if ((hits[z][j].y < upper_y) & (hits[z][j].y > lower_y) & (not hits[z][j].used) & (not indicator)):

                crossing_points.append(hits[z][j].index)
                Z.append(z)
                Y.append(hits[z][j].y)
                weights.append(1 / (hits[z][j].dist2Wire)**(0.5))
                marks[z].append(j)
                indicator = True

        if indicator:
            n += 1


    if n < n_min:

        return 0, crossing_points, [0., 0.]


    else:

        lin_regr = np.polyfit(Z, Y, 1, w = weights)
        dists_0 = []
        dists_1 = []
        
        for j in crossing_points:
            
            dists_0.append(dist2Track((event.Wz[j], event.Wy[j]), lin_regr)/event.dist2Wire[j])
            dists_1.append(np.abs(dist2Track((event.Wz[j], event.Wy[j]), lin_regr)-event.dist2Wire[j]))

        if clf.predict([n, np.max(dists_0), np.min(dists_0), np.average(dists_0), np.max(dists_1), np.min(dists_1), np.average(dists_1)])[0]==0:
            
            return 0, crossing_points, [0., 0.]
        
        else:
            
            for z in hits:

                for i in marks[z]:

                    hits[z][i].used = True

            return 1, crossing_points, lin_regr



def crossing_lines(k1, b1, k2, b2):

    z = (b2 - b1) / (k1 - k2)
    y = z * k1 + b1

    return (y, z)



def loop_yz_new(event, n_min, plane_width, ind, clf):
    """
    Finds all possible candidates for being tracks in 2d-space (z, y). Algorithm uses only hits from Y-views. For all 
    hits in the first plane and for all hits in the last plane it constructs lines using all possible pairs 
    of points(point from the 1st plane, point from the last plane). All this lines are supplied to points_crossing_line_yz().
    
    Args:
        event: pd.DataFrame() with all hits of any event;
        n_min: minimal number of points intercepting track for recognition this track;
        plane_width: vertical window of finding line;
        ind: 0 = 1 & 2 stations, 1 = 3 & 4 stations.
    Returns:
        tracks, linking_table
        tracks: dictionary of all recognised lines in 2d space (y, z), key = id of track, value = (k, b);
        linking_table: links each track from tracks and his hits, represented by dictionary:
            key = id of track, value = array of indexes of his hits.
    """

    hits = conventor_yz(event, ind) # dictionary with hits: key = z; value = array of objects with fields(y, dist2Wire, index)


    if (ind):

        start_zs = [3321.15, 3322.25]
        end_zs = [3553.75, 3554.85]
        event = modify_for_yz_analysis_3_4(event)

    else:

        start_zs = [2581.15, 2582.25]
        end_zs = [2813.75, 2814.85]
        event = modify_for_yz_analysis_1_2(event)


    tracks = {} #finded tracks: key = id of recognized track; value = (k, p)
    linking_table = {} # key = id of recognized track; value = array of hit ID's from the main table
    trackID = 1

    for n in range(16, n_min - 1, -1):
    
        for start_z in (set(start_zs) & set(hits.keys())):

            for i in hits[start_z]:

                for end_z in (set(end_zs) & set(hits.keys())):

                    for j in hits[end_z]:
                        
                        if ((not i.used) & (not j.used)):

                            k, b = get_plane((i.y, start_z), (j.y, end_z))

                            indicator, crossing_points, lin_regr = points_crossing_line_yz_new(k, b, plane_width, hits, n, clf, event)

                            if indicator == 1:
                                tracks[trackID] = lin_regr
                                linking_table[trackID] = crossing_points
                                trackID += 1

    return remove_unnecessary_yz(tracks, linking_table)



def remove_unnecessary_yz(tracks, linking_table):
    """
    Rejects tracks that leave volume of installation.
    
    Args:
        tracks: candidates to track;
        linking_table: links tracks with hits.
    Returns:
        tracks, linking_table without rejected tracks.
    """

    tracks_for_remove = []

    for i in tracks:

        y = tracks[i][0] * 3000 + tracks[i][1]

        if ((y < -500) | (y > 500)):
            tracks_for_remove.append(i)


    for i in tracks_for_remove:

        tracks.pop(i, None)
        linking_table.pop(i, None)

        
    return tracks, linking_table