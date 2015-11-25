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

#return k, b from equation y = k * x + b; input = two points
def get_plane(point1, point2):
    y1 = point1[0]
    z1 = point1[1]
    y2 = point2[0]
    z2 = point2[1]
    
    k = float(y2 - y1)/float(z2 - z1)
    b = y1 - k*z1
    return k, b

#select Y-views before magnet and add to dataframe 2 columns: Wz, Wy  - coordinates of centre of the tube
def modify_for_yz_analisys(event):    
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

#input: pd.DataFrame(); output: dictionary with key = Wz
def conventor_yz(event):
    event = modify_for_yz_analisys(event)
    dictionary = {}
    for i in event.index:
        dictionary.setdefault(event.Wz[i], []).append(ParametresYZ(event.Wy[i], event.dist2Wire[i], event.Index[i]))
    return dictionary

#for known k and b find ID's of hits crossing this line; also it returns linnear regression on this hits
def points_crossing_line_yz(plane_k, plane_b, plane_width, hits, n_min):
    lower_y = 0.
    upper_y = 0.
    crossing_points = []
    Y = [] # for linnear regression
    Z = []
    n = 0 # number of touched layers
    for i in hits:
        lower_y = plane_k * i + plane_b - 1. * plane_width
        upper_y = plane_k * i + plane_b + 1. * plane_width
        indicator = False
        for j in hits[i]:
            if ((j.y < upper_y) & (j.y > lower_y)):
                crossing_points.append(j.index)
                Z.append(i)
                Y.append(j.y)
                indicator = True
        if indicator:
            n += 1
    if n < n_min:
        return 0, crossing_points, [0., 0.]
    else:
        lin_regr = np.polyfit(Z, Y, 1)
        return 1, crossing_points, lin_regr

def loop_yz(event, n_min, plane_width):
    hits = conventor_yz(event) # dictionary with hits: key = z; value = array of objects with fields(y, dist2Wire, index)
    tracks = {} #finded tracks: key = id of recognized track; value = (k, p)
    linking_table = {} # key = id of recognized track; value = array of hit ID's from the main table
    trackID = 1
    start_z = [2581.1500000000001, 2582.25]
    end_z = [2813.75, 2814.85]
    for start_key in (set(start_z) & set(hits.keys())):
        for i in hits[start_key]:
            for end_key in (set(end_z) & set(hits.keys())):
                for j in hits[end_key]:
                    k, b = get_plane((i.y, start_key), (j.y, end_key))
                    indicator, crossing_points, lin_regr = points_crossing_line_yz(k, b, plane_width, hits, n_min)
                    if indicator == 1:
                        tracks[trackID] = lin_regr
                        linking_table[trackID] = crossing_points
                        trackID += 1
    return tracks, linking_table

def crossing_lines(k1, b1, k2, b2):
    z = (b2 - b1) / (k1 - k2)
    y = z * k1 + b1
    return (y, z)

def remove_unnecessary_yz(tracks, linking_table):
    tracks_for_remove = []
    for i in tracks:
        y = tracks[i][0] * 3000 + tracks[i][1]
        if ((y < -500) | (y > 500)):
            tracks_for_remove.append(i)
    for i in tracks_for_remove:
        tracks.pop(i, None)
        linking_table.pop(i, None)
    return tracks, linking_table
