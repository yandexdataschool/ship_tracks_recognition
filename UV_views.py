import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

class parametresXZ:
    def __init__(self, dist2Centre, dist2Wire, angle, x, used):
        self.dist2Centre = dist2Centre
        self.dist2Wire = dist2Wire
        self.angle = angle
        self.used = used
        self.x = x


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

def modify_for_xz_analysis(event):
    """
    Gets table of hits, fetchs only hits from U,V-views and add to it columns 'Wz', 'dist2Centre' - z-coordinates of centres of 
    tubes and distances from tubes to point (0, 0).
    
    Args:
        event: pd.DataFrame() that necesserily contains this columns: 
        'StatNb' - number of station,
        'ViewNb' - number of view,
        'PlaneNb' - number of plane,
        'LayerNb' - number of layer,
        'StrawNb' - number of straw.
        
    Returns:
        the same pd.DataFrame() but with 2 new columns: 'Wz', 'dist2Centre'
    """   
    layer1100 = selector(event, 1, 1, 0, 0)
    layer1101 = selector(event, 1, 1, 0, 1)
    layer1110 = selector(event, 1, 1, 1, 0)
    layer1111 = selector(event, 1, 1, 1, 1)

    layer1200 = selector(event, 1, 2, 0, 0)
    layer1201 = selector(event, 1, 2, 0, 1)
    layer1210 = selector(event, 1, 2, 1, 0)
    layer1211 = selector(event, 1, 2, 1, 1)

    layer2100 = selector(event, 2, 1, 0, 0)
    layer2101 = selector(event, 2, 1, 0, 1)
    layer2110 = selector(event, 2, 1, 1, 0)
    layer2111 = selector(event, 2, 1, 1, 1)

    layer2200 = selector(event, 2, 2, 0, 0)
    layer2201 = selector(event, 2, 2, 0, 1)
    layer2210 = selector(event, 2, 2, 1, 0)
    layer2211 = selector(event, 2, 2, 1, 1)
    
    #1-u
    z1100 = 2598. - 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1100.loc[:, 'dist2Centre'] = -(0.5*0.9828 + 1.76*(layer1100.StrawNb.values-1)) + 499
    layer1100.loc[:, 'Wz'] = np.ones(layer1100.shape[0])*z1100

    z1101 = z1100 + 1.1
    layer1101.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.88 + 1.76*(layer1101.StrawNb.values-1)) + 499
    layer1101.loc[:, 'Wz'] = np.ones(layer1101.shape[0])*z1101

    z1110 = z1100 + 2.6
    layer1110.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 + 1.76*(layer1110.StrawNb.values-1)) + 499
    layer1110.loc[:, 'Wz'] = np.ones(layer1110.shape[0])*z1110

    z1111 = z1110 + 1.1
    layer1111.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1111.StrawNb.values-1)) + 499
    layer1111.loc[:, 'Wz'] = np.ones(layer1111.shape[0])*z1111

    #1-v
    z1200 = 2598. + 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1200.loc[:, 'dist2Centre'] = -(0.5*0.9828 + 1.76*(layer1200.StrawNb.values-1)) + 499
    layer1200.loc[:, 'Wz'] = np.ones(layer1200.shape[0])*z1200
    
    z1201 = z1200 + 1.1
    layer1201.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.88 + 1.76*(layer1201.StrawNb.values-1)) + 499
    layer1201.loc[:, 'Wz'] = np.ones(layer1201.shape[0])*z1201

    z1210 = z1200 + 2.6
    layer1210.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 + 1.76*(layer1210.StrawNb.values-1)) + 499
    layer1210.loc[:, 'Wz'] = np.ones(layer1210.shape[0])*z1210

    z1211 = z1210 + 1.1
    layer1211.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1211.StrawNb.values-1)) + 499
    layer1211.loc[:, 'Wz'] = np.ones(layer1211.shape[0])*z1211

    #2-u
    z2100 = 2798. - 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2100.loc[:, 'dist2Centre'] = -(0.5*0.9828 + 1.76*(layer2100.StrawNb.values-1)) + 499
    layer2100.loc[:, 'Wz'] = np.ones(layer2100.shape[0])*z2100

    z2101 = z2100 + 1.1
    layer2101.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.88 + 1.76*(layer2101.StrawNb.values-1)) + 499
    layer2101.loc[:, 'Wz'] = np.ones(layer2101.shape[0])*z2101

    z2110 = z2100 + 2.6
    layer2110.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 + 1.76*(layer2110.StrawNb.values-1)) + 499
    layer2110.loc[:, 'Wz'] = np.ones(layer2110.shape[0])*z2110

    z2111 = z2110 + 1.1
    layer2111.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2111.StrawNb.values-1)) + 499
    layer2111.loc[:, 'Wz'] = np.ones(layer2111.shape[0])*z2111

    #2-v
    z2200 = 2798. + 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2200.loc[:, 'dist2Centre'] = -(0.5*0.9828 + 1.76*(layer2200.StrawNb.values-1)) + 499
    layer2200.loc[:, 'Wz'] = np.ones(layer2200.shape[0])*z2200

    z2201 = z2200 + 1.1
    layer2201.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.88 + 1.76*(layer2201.StrawNb.values-1)) + 499
    layer2201.loc[:, 'Wz'] = np.ones(layer2201.shape[0])*z2201

    z2210 = z2200 + 2.6
    layer2210.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 + 1.76*(layer2210.StrawNb.values-1)) + 499
    layer2210.loc[:, 'Wz'] = np.ones(layer2210.shape[0])*z2210

    z2211 = z2210 + 1.1
    layer2211.loc[:, 'dist2Centre'] = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2211.StrawNb.values-1)) + 499
    layer2211.loc[:, 'Wz'] = np.ones(layer2211.shape[0])*z2211
    
    layers = [layer1100, layer1101, layer1110, layer1111,
              layer1200, layer1201, layer1210, layer1211,
              layer2100, layer2101, layer2110, layer2111,
              layer2200, layer2201, layer2210, layer2211]

    return pd.concat(layers, axis=0)

def conventor_xz(event):
    """
    Gets pd.DataFrame() and transforms it into dictionary.
    
    Args:
        event: pd.DataFrame() that necesserily contains columns 'Wz', 'dist2Centre', 'dist2Wire', 'ViewNb'.
    Returns:
        dictionary: keys are values of 'Wz'; values are stuctures with fields(dist2Centre, dist2Wire, angle, x, index, used).
    """
    event = modify_for_xz_analysis(event)
    dictionary = {}
    for i in event.index:
        dictionary.setdefault(event.Wz[i], {})[i] = parametresXZ(event.dist2Centre[i], event.dist2Wire[i], (event.ViewNb[i] - 1.5) * 2 * np.pi / 36, 50000, False)
    return dictionary

def points_crossing_line_xz(k, b, width, hits, intersecting_hits, n_min):
    """
    Counts the number of points which intercept line with parametres: plane_k, plane_b, plane_width.
    If the result more than n_min than makes linnear regression on this points.
    
    Args:
        plane_k, plane_b, plane_width: parametres of line in 2d space (z, x). It's a hyperplane in 3d space;
        hits: dictionary that contains all hits, key is z coordinate of centre of tube, value is another
            dictionary with key = index, value = structure with fields-dist2Centre, dist2Wire, angle, used;
        n_min: minimal number of hits intercepting a track;
        intersecting_hits: dictionary containing subset of indexes of hits wich are intersected by the line in 2d-space (z, y).
    Returns:
        indicator, crossing_points, lin_regr;
        indicator: false- means line with this parametres doesn't cover a track,
            true- line covers a track;
        crossing_points: array of indexes of points that determine the track;
        lin_regr: parametres k, b of linnear regression on crossing points.
    """
    lower_x = 0.
    upper_x = 0.
    crossing_points = []
    X = [] # for linnear regression
    Z = []
    n = 0 # number of touched layers
    marks = {}
    for z in intersecting_hits:
        marks[z] = []
        lower_x = k * z + b - 1. * width
        upper_x = k * z + b + 1. * width
        indicator = False
        for j in intersecting_hits[z]:
            if ((hits[z][j].x < upper_x) & (hits[z][j].x > lower_x) & (not hits[z][j].used) & (not indicator)):
                crossing_points.append(j)
                Z.append(z)
                X.append(hits[z][j].x)
                marks[z].append(j)
                indicator = True
        if indicator: n += 1
    if n < n_min:
        return 0, crossing_points, [0., 0.]
    else:
        lin_regr = np.polyfit(Z, X, 1)
        for z in marks:
            for i in marks[z]:
                hits[z][i].used = True
        return 1, crossing_points, lin_regr

def loop_xz(event, tracks, linking_table, n_min, width):
    """
    Gets tracks and linking_table received from previous stage Y-views analysis. Fetches only tracks which intersect 
    more than n_min hits in 2d-space (z, x). Every track may be rejected or give 1 and more tracks in 2d-space (z, x).
    
    Args:
        event: pd.DataFrame() with all hits of any event;
        n_min: minimal number of points intercepting track for recognition this track;
        plane_width: stereo window of finding line;
        tracks: all tracks recognized on previous stage;
        linking_table: table from previous stage.
    Returns:
        new_tracks: new dictionary, key in new_tracks = "key from 'tracks'" * 10000 + new key, for cases when one track from
            'tracks' contains 2 and more tracks in 3d-space;
        new_linking_table: links each track from new_tracks and his hits, represented by dictionary:
            key = id of track, value = array of indexes of his hits.
    """
    hits = conventor_xz(event)
    new_linking_table = {}
    new_tracks = {}
    x_coordinates = {}
    new_trackID = 1
    start_zs = [2591.2793, 2592.3793000000001]
    end_zs = [2806.0792999999999, 2804.9793]
    for track_id in tracks:
        intersecting_hits = {}
        k = tracks[track_id][0]
        b = tracks[track_id][1]
        n = 0
        for z in hits:
            y = k * z + b
            for hit_index in hits[z]:
                x = hits[z][hit_index].dist2Centre * sin(hits[z][hit_index].angle) + (hits[z][hit_index].dist2Centre * cos(hits[z][hit_index].angle) - y) / tan(hits[z][hit_index].angle)
                if ((x > -250) & (x < 250) & (not hits[z][hit_index].used)):
                    hits[z][hit_index].x = x
                    x_coordinates[hit_index] = x
                    intersecting_hits.setdefault(z, []).append(hit_index)
                    n += 1
        if (n >= n_min):
            for start_z in (set(start_zs) & set(intersecting_hits)):
                for i in intersecting_hits[start_z]:
                    for end_z in (set(end_zs) & set(intersecting_hits)):
                        for j in intersecting_hits[end_z]:
                            if ((not hits[start_z][i].used) & (not hits[end_z][j].used)):
                                new_k, new_b = get_plane((hits[start_z][i].x, start_z), (hits[end_z][j].x, end_z))
                                indicator, crossing_points, lin_regr = points_crossing_line_xz(new_k, new_b, width, hits, intersecting_hits, n_min)
                                if indicator == 1:
                                    new_tracks[track_id * 10000 + new_trackID] = lin_regr
                                    new_linking_table[track_id * 10000 + new_trackID] = crossing_points
                                    new_trackID += 1
    return new_tracks, new_linking_table, x_coordinates