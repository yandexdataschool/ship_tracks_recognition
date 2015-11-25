import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from math import *

class Layer:
    def __init__(self, z, bias):
        self.bias = bias
        self.z = z
    def y(self, straw_id):
        return 499 - self.bias - straw_id * 1.76


class ParametresYZ:
    def __init__(self, y, dist2Wire, index):
        self.dist2Wire = dist2Wire
        self.y = y
        self.index = index


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

def modify_for_yz_analisys(event):
    """
    Gets table of hits, fetchs only hits from Y-views and add to it columns 'Wz', 'Wy' - coordinates of centres of 
    tubes in plane (z, y).
    
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
    layer1000.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer1000.StrawNb.values-1)) + 499
    layer1000.loc[:, 'Wz'] = np.ones(layer1000.shape[0])*z1000

    z1001 = z1000 + 1.1
    layer1001.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer1001.StrawNb.values-1)) + 499
    layer1001.loc[:, 'Wz'] = np.ones(layer1001.shape[0])*z1001

    z1010 = z1000 + 2.6
    layer1010.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer1010.StrawNb.values-1)) + 499
    layer1010.loc[:, 'Wz'] = np.ones(layer1010.shape[0])*z1010

    z1011 = z1010 + 1.1
    layer1011.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1011.StrawNb.values-1)) + 499
    layer1011.loc[:, 'Wz'] = np.ones(layer1011.shape[0])*z1011

    #1-y2
    z1300 = 2598. + 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1300.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer1300.StrawNb.values-1)) + 499
    layer1300.loc[:, 'Wz'] = np.ones(layer1300.shape[0])*z1300

    z1301 = z1300 + 1.1
    layer1301.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer1301.StrawNb.values-1)) + 499
    layer1301.loc[:, 'Wz'] = np.ones(layer1301.shape[0])*z1301

    z1310 = z1300 + 2.6
    layer1310.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer1310.StrawNb.values-1)) + 499
    layer1310.loc[:, 'Wz'] = np.ones(layer1310.shape[0])*z1310

    z1311 = z1310 + 1.1
    layer1311.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1311.StrawNb.values-1)) + 499
    layer1311.loc[:, 'Wz'] = np.ones(layer1311.shape[0])*z1311

    #2-y1
    z2000 = 2798. - 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2000.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer2000.StrawNb.values-1)) + 499
    layer2000.loc[:, 'Wz'] = np.ones(layer2000.shape[0])*z2000

    z2001 = z2000 + 1.1
    layer2001.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer2001.StrawNb.values-1)) + 499
    layer2001.loc[:, 'Wz'] = np.ones(layer2001.shape[0])*z2001

    z2010 = z2000 + 2.6
    layer2010.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer2010.StrawNb.values-1)) + 499
    layer2010.loc[:, 'Wz'] = np.ones(layer2010.shape[0])*z2010

    z2011 = z2010 + 1.1
    layer2011.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2011.StrawNb.values-1)) + 499
    layer2011.loc[:, 'Wz'] = np.ones(layer2011.shape[0])*z2011

    #2-y2
    z2300 = 2798. + 15. - 0.5*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2300.loc[:, 'Wy'] = -(0.5*0.9828 + 1.76*(layer2300.StrawNb.values-1)) + 499
    layer2300.loc[:, 'Wz'] = np.ones(layer2300.shape[0])*z2300

    z2301 = z2300 + 1.1
    layer2301.loc[:, 'Wy'] = -(0.5*0.9828 - 0.88 + 1.76*(layer2301.StrawNb.values-1)) + 499
    layer2301.loc[:, 'Wz'] = np.ones(layer2301.shape[0])*z2301

    z2310 = z2300 + 2.6
    layer2310.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 + 1.76*(layer2310.StrawNb.values-1)) + 499
    layer2310.loc[:, 'Wz'] = np.ones(layer2310.shape[0])*z2310

    z2311 = z2310 + 1.1
    layer2311.loc[:, 'Wy'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2311.StrawNb.values-1)) + 499
    layer2311.loc[:, 'Wz'] = np.ones(layer2311.shape[0])*z2311
    
    layers = [layer1000, layer1001, layer1010, layer1011,
              layer1300, layer1301, layer1310, layer1311,
              layer2000, layer2001, layer2010, layer2011,
              layer2300, layer2301, layer2310, layer2311]

    return pd.concat(layers, axis=0)

def conventor_yz(event):
    """
    Gets pd.DataFrame() and transforms it into dictionary.
    
    Args:
        event: pd.DataFrame() that necesserily contains columns 'Wz', 'Wy', 'dist2Wire'.
    Returns:
        dictionary: keys are values of 'Wz'; values are stuctures with fields(y, dist2Wire, index)
    """
    event = modify_for_yz_analisys(event)
    dictionary = {}
    for i in event.index:
        dictionary.setdefault(event.Wz[i], []).append(ParametresYZ(event.Wy[i], event.dist2Wire[i], event.Index[i]))
    return dictionary

def points_crossing_line_yz(plane_k, plane_b, plane_width, hits, n_min):
    """
    Counts the number of points which intercept line with parametres: plane_k, plane_b, plane_width.
    If the result more than n_min than makes linnear regression on this points.
    
    Args:
        plane_k, plane_b, plane_width: parametres of line in 2d space (y, z). It's a hyperplane in 3d space;
        hits: dictionary that contains all hits, key is z coordinate of centre of tube, value is structure 
            with fields- y, dist2Wire, index;
        n_min: minimal number of hits intercepting a track.
    Returns:
        indicator, crossing_points, lin_regr;
        indicator: false- means line with this parametres doesn't cover a track,
            true- line covers a track;
        crossing_points: array of indexes of points that determine the track;
        lin_regr: parametres k, b of linnear regression on crossing points. 
    """
    lower_y = 0.
    upper_y = 0.
    crossing_points = []
    Y = [] # for linnear regression
    Z = []
    n = 0 # number of touched layers
    for z in hits:
        lower_y = plane_k * z + plane_b - 1. * plane_width
        upper_y = plane_k * z + plane_b + 1. * plane_width
        indicator = False
        for j in hits[z]:
            if ((j.y < upper_y) & (j.y > lower_y)):
                crossing_points.append(j.index)
                Z.append(z)
                Y.append(j.y)
                indicator = True
        if indicator:
            n += 1
    if n < n_min:
        return 0, crossing_points, [0., 0.]
    else:
        lin_regr = np.polyfit(Z, Y, 1)
        return 1, crossing_points, lin_regr
    
def crossing_lines(k1, b1, k2, b2):
    z = (b2 - b1) / (k1 - k2)
    y = z * k1 + b1
    return (y, z)

def loop_yz(event, n_min, plane_width):
    """
    Finds all possible candidates for being tracks in 2d-space (z, y). Algorithm uses only hits from Y-views. For all 
    hits in the first plane and for all hits in the last plane it constructs lines using all possible pairs 
    of points(point from the 1st plane, point from the last plane). All this line are supplied to points_crossing_line_yz().
    
    Args:
        event: pd.DataFrame() with all hits of any event;
        n_min: minimal number of points intercepting track for recognition this track;
        plane_width: vertical window of finding line.
    Returns:
        tracks, linking_table
        tracks: dictionary of all recognised lines in 2d space (y, z), key = id of track, value = (k, b);
        linking_table: links each track from tracks and his hits, represented by dictionary:
            key = id of track, value = array of indexes of his hits.
    """
    hits = conventor_yz(event) # dictionary with hits: key = z; value = array of objects with fields(y, dist2Wire, index)
    tracks = {} #finded tracks: key = id of recognized track; value = (k, p)
    linking_table = {} # key = id of recognized track; value = array of hit ID's from the main table
    trackID = 1
    start_zs = [2581.1500000000001, 2582.25]
    end_zs = [2813.75, 2814.85]
    for start_z in (set(start_zs) & set(hits.keys())):
        for i in hits[start_z]:
            for end_z in (set(end_zs) & set(hits.keys())):
                for j in hits[end_z]:
                    k, b = get_plane((i.y, start_z), (j.y, end_z))
                    indicator, crossing_points, lin_regr = points_crossing_line_yz(k, b, plane_width, hits, n_min)
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
