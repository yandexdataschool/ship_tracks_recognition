__author__ = 'Alenkin Oleg, Mikhail Hushchyn'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

class parametresXZ:

    def __init__(self, Wu, dist2Wire, angle, x, used):
        self.Wu = Wu
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


def modify_for_xz_analysis_1_2(event):
    """
    Gets table of hits, fetchs only hits from U,V-views (1 & 2 stations) and add to it columns 'Wz', 'Wu' - 
    z-coordinates of centres of tubes and distances from tubes to point (0, 0).
    
    Args:
        event: pd.DataFrame() that necesserily contains this columns: 
        'StatNb' - number of station,
        'ViewNb' - number of view,
        'PlaneNb' - number of plane,
        'LayerNb' - number of layer,
        'StrawNb' - number of straw.
        
    Returns:
        the same pd.DataFrame() but with 2 new columns: 'Wz', 'Wu'
    """   
    layer1100 = selector(event, 1, 1, 0, 0)#
    layer1101 = selector(event, 1, 1, 0, 1)#!
    layer1110 = selector(event, 1, 1, 1, 0)#!
    layer1111 = selector(event, 1, 1, 1, 1)#

    layer1200 = selector(event, 1, 2, 0, 0)#
    layer1201 = selector(event, 1, 2, 0, 1)#
    layer1210 = selector(event, 1, 2, 1, 0)#
    layer1211 = selector(event, 1, 2, 1, 1)#

    layer2100 = selector(event, 2, 1, 0, 0)#
    layer2101 = selector(event, 2, 1, 0, 1)#
    layer2110 = selector(event, 2, 1, 1, 0)#
    layer2111 = selector(event, 2, 1, 1, 1)#

    layer2200 = selector(event, 2, 2, 0, 0)#
    layer2201 = selector(event, 2, 2, 0, 1)#
    layer2210 = selector(event, 2, 2, 1, 0)#
    layer2211 = selector(event, 2, 2, 1, 1)#

    sign1 = -1.
    sign2 = +1.
    
    #1-u
    z1100 = 2598. - 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1100_wu = -(0.5*0.9828 + 1.76*(layer1100.StrawNb.values-1)) + 499 + 0.44
    layer1100_wv1 = -250. * np.ones(len(layer1100))
    layer1100_wv2 = 250. * np.ones(len(layer1100))
    layer1100_wx1 = layer1100_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer1100_wu * np.sin(5. * np.pi / 180.)
    layer1100_wx2 = layer1100_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer1100_wu * np.sin(5. * np.pi / 180.)
    layer1100_wy1 = -sign1 * layer1100_wv1 * np.sin(5. * np.pi / 180.) + layer1100_wu * np.cos(5. * np.pi / 180.)
    layer1100_wy2 = -sign1 * layer1100_wv2 * np.sin(5. * np.pi / 180.) + layer1100_wu * np.cos(5. * np.pi / 180.)
    layer1100_wz = np.ones(layer1100.shape[0])*z1100

    z1101 = z1100 + 1.1
    layer1101_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer1101.StrawNb.values-1)) + 499 + 0.44
    layer1101_wv1 = -250. * np.ones(len(layer1101))
    layer1101_wv2 = 250. * np.ones(len(layer1101))
    layer1101_wx1 = layer1101_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer1101_wu * np.sin(5. * np.pi / 180.)
    layer1101_wx2 = layer1101_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer1101_wu * np.sin(5. * np.pi / 180.)
    layer1101_wy1 = -sign1 * layer1101_wv1 * np.sin(5. * np.pi / 180.) + layer1101_wu * np.cos(5. * np.pi / 180.)
    layer1101_wy2 = -sign1 * layer1101_wv2 * np.sin(5. * np.pi / 180.) + layer1101_wu * np.cos(5. * np.pi / 180.)
    layer1101_wz = np.ones(layer1101.shape[0])*z1101

    z1110 = z1100 + 2.6
    layer1110_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer1110.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1110_wv1 = -250. * np.ones(len(layer1110))
    layer1110_wv2 = 250. * np.ones(len(layer1110))
    layer1110_wx1 = layer1110_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer1110_wu * np.sin(5. * np.pi / 180.)
    layer1110_wx2 = layer1110_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer1110_wu * np.sin(5. * np.pi / 180.)
    layer1110_wy1 = -sign1 * layer1110_wv1 * np.sin(5. * np.pi / 180.) + layer1110_wu * np.cos(5. * np.pi / 180.)
    layer1110_wy2 = -sign1 * layer1110_wv2 * np.sin(5. * np.pi / 180.) + layer1110_wu * np.cos(5. * np.pi / 180.)
    layer1110_wz = np.ones(layer1110.shape[0])*z1110

    z1111 = z1110 + 1.1
    layer1111_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1111.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1111_wv1 = -250. * np.ones(len(layer1111))
    layer1111_wv2 = 250. * np.ones(len(layer1111))
    layer1111_wx1 = layer1111_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer1111_wu * np.sin(5. * np.pi / 180.)
    layer1111_wx2 = layer1111_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer1111_wu * np.sin(5. * np.pi / 180.)
    layer1111_wy1 = -sign1 * layer1111_wv1 * np.sin(5. * np.pi / 180.) + layer1111_wu * np.cos(5. * np.pi / 180.)
    layer1111_wy2 = -sign1 * layer1111_wv2 * np.sin(5. * np.pi / 180.) + layer1111_wu * np.cos(5. * np.pi / 180.)
    layer1111_wz = np.ones(layer1111.shape[0])*z1111

    #1-v
    z1200 = 2598. + 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer1200_wu = -(0.5*0.9828 + 1.76*(layer1200.StrawNb.values-1)) + 499 + 0.44
    layer1200_wv1 = -250. * np.ones(len(layer1200))
    layer1200_wv2 = 250. * np.ones(len(layer1200))
    layer1200_wx1 = layer1200_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer1200_wu * np.sin(5. * np.pi / 180.)
    layer1200_wx2 = layer1200_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer1200_wu * np.sin(5. * np.pi / 180.)
    layer1200_wy1 = -sign2 * layer1200_wv1 * np.sin(5. * np.pi / 180.) + layer1200_wu * np.cos(5. * np.pi / 180.)
    layer1200_wy2 = -sign2 * layer1200_wv2 * np.sin(5. * np.pi / 180.) + layer1200_wu * np.cos(5. * np.pi / 180.)
    layer1200_wz = np.ones(layer1200.shape[0])*z1200
    
    z1201 = z1200 + 1.1
    layer1201_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer1201.StrawNb.values-1)) + 499 + 0.44
    layer1201_wv1 = -250. * np.ones(len(layer1201))
    layer1201_wv2 = 250. * np.ones(len(layer1201))
    layer1201_wx1 = layer1201_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer1201_wu * np.sin(5. * np.pi / 180.)
    layer1201_wx2 = layer1201_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer1201_wu * np.sin(5. * np.pi / 180.)
    layer1201_wy1 = -sign2 * layer1201_wv1 * np.sin(5. * np.pi / 180.) + layer1201_wu * np.cos(5. * np.pi / 180.)
    layer1201_wy2 = -sign2 * layer1201_wv2 * np.sin(5. * np.pi / 180.) + layer1201_wu * np.cos(5. * np.pi / 180.)
    layer1201_wz = np.ones(layer1201.shape[0])*z1201

    z1210 = z1200 + 2.6
    layer1210_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer1210.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1210_wv1 = -250. * np.ones(len(layer1210))
    layer1210_wv2 = 250. * np.ones(len(layer1210))
    layer1210_wx1 = layer1210_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer1210_wu * np.sin(5. * np.pi / 180.)
    layer1210_wx2 = layer1210_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer1210_wu * np.sin(5. * np.pi / 180.)
    layer1210_wy1 = -sign2 * layer1210_wv1 * np.sin(5. * np.pi / 180.) + layer1210_wu * np.cos(5. * np.pi / 180.)
    layer1210_wy2 = -sign2 * layer1210_wv2 * np.sin(5. * np.pi / 180.) + layer1210_wu * np.cos(5. * np.pi / 180.)
    layer1210_wz = np.ones(layer1210.shape[0])*z1210

    z1211 = z1210 + 1.1
    layer1211_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer1211.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer1211_wv1 = -250. * np.ones(len(layer1211))
    layer1211_wv2 = 250. * np.ones(len(layer1211))
    layer1211_wx1 = layer1211_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer1211_wu * np.sin(5. * np.pi / 180.)
    layer1211_wx2 = layer1211_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer1211_wu * np.sin(5. * np.pi / 180.)
    layer1211_wy1 = -sign2 * layer1211_wv1 * np.sin(5. * np.pi / 180.) + layer1211_wu * np.cos(5. * np.pi / 180.)
    layer1211_wy2 = -sign2 * layer1211_wv2 * np.sin(5. * np.pi / 180.) + layer1211_wu * np.cos(5. * np.pi / 180.)
    layer1211_wz = np.ones(layer1211.shape[0])*z1211

    #2-u
    z2100 = 2798. - 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2100_wu = -(0.5*0.9828 + 1.76*(layer2100.StrawNb.values-1)) + 499 + 0.44
    layer2100_wv1 = -250. * np.ones(len(layer2100))
    layer2100_wv2 = 250. * np.ones(len(layer2100))
    layer2100_wx1 = layer2100_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer2100_wu * np.sin(5. * np.pi / 180.)
    layer2100_wx2 = layer2100_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer2100_wu * np.sin(5. * np.pi / 180.)
    layer2100_wy1 = -sign1 * layer2100_wv1 * np.sin(5. * np.pi / 180.) + layer2100_wu * np.cos(5. * np.pi / 180.)
    layer2100_wy2 = -sign1 * layer2100_wv2 * np.sin(5. * np.pi / 180.) + layer2100_wu * np.cos(5. * np.pi / 180.)
    layer2100_wz = np.ones(layer2100.shape[0])*z2100

    z2101 = z2100 + 1.1
    layer2101_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer2101.StrawNb.values-1)) + 499 + 0.44
    layer2101_wv1 = -250. * np.ones(len(layer2101))
    layer2101_wv2 = 250. * np.ones(len(layer2101))
    layer2101_wx1 = layer2101_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer2101_wu * np.sin(5. * np.pi / 180.)
    layer2101_wx2 = layer2101_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer2101_wu * np.sin(5. * np.pi / 180.)
    layer2101_wy1 = -sign1 * layer2101_wv1 * np.sin(5. * np.pi / 180.) + layer2101_wu * np.cos(5. * np.pi / 180.)
    layer2101_wy2 = -sign1 * layer2101_wv2 * np.sin(5. * np.pi / 180.) + layer2101_wu * np.cos(5. * np.pi / 180.)
    layer2101_wz = np.ones(layer2101.shape[0])*z2101

    z2110 = z2100 + 2.6
    layer2110_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer2110.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2110_wv1 = -250. * np.ones(len(layer2110))
    layer2110_wv2 = 250. * np.ones(len(layer2110))
    layer2110_wx1 = layer2110_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer2110_wu * np.sin(5. * np.pi / 180.)
    layer2110_wx2 = layer2110_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer2110_wu * np.sin(5. * np.pi / 180.)
    layer2110_wy1 = -sign1 * layer2110_wv1 * np.sin(5. * np.pi / 180.) + layer2110_wu * np.cos(5. * np.pi / 180.)
    layer2110_wy2 = -sign1 * layer2110_wv2 * np.sin(5. * np.pi / 180.) + layer2110_wu * np.cos(5. * np.pi / 180.)
    layer2110_wz = np.ones(layer2110.shape[0])*z2110

    z2111 = z2110 + 1.1
    layer2111_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2111.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2111_wv1 = -250. * np.ones(len(layer2111))
    layer2111_wv2 = 250. * np.ones(len(layer2111))
    layer2111_wx1 = layer2111_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer2111_wu * np.sin(5. * np.pi / 180.)
    layer2111_wx2 = layer2111_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer2111_wu * np.sin(5. * np.pi / 180.)
    layer2111_wy1 = -sign1 * layer2111_wv1 * np.sin(5. * np.pi / 180.) + layer2111_wu * np.cos(5. * np.pi / 180.)
    layer2111_wy2 = -sign1 * layer2111_wv2 * np.sin(5. * np.pi / 180.) + layer2111_wu * np.cos(5. * np.pi / 180.)
    layer2111_wz = np.ones(layer2111.shape[0])*z2111

    #2-v
    z2200 = 2798. + 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer2200_wu = -(0.5*0.9828 + 1.76*(layer2200.StrawNb.values-1)) + 499 + 0.44
    layer2200_wv1 = -250. * np.ones(len(layer2200))
    layer2200_wv2 = 250. * np.ones(len(layer2200))
    layer2200_wx1 = layer2200_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer2200_wu * np.sin(5. * np.pi / 180.)
    layer2200_wx2 = layer2200_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer2200_wu * np.sin(5. * np.pi / 180.)
    layer2200_wy1 = -sign2 * layer2200_wv1 * np.sin(5. * np.pi / 180.) + layer2200_wu * np.cos(5. * np.pi / 180.)
    layer2200_wy2 = -sign2 * layer2200_wv2 * np.sin(5. * np.pi / 180.) + layer2200_wu * np.cos(5. * np.pi / 180.)
    layer2200_wz = np.ones(layer2200.shape[0])*z2200

    z2201 = z2200 + 1.1
    layer2201_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer2201.StrawNb.values-1)) + 499 + 0.44
    layer2201_wv1 = -250. * np.ones(len(layer2201))
    layer2201_wv2 = 250. * np.ones(len(layer2201))
    layer2201_wx1 = layer2201_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer2201_wu * np.sin(5. * np.pi / 180.)
    layer2201_wx2 = layer2201_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer2201_wu * np.sin(5. * np.pi / 180.)
    layer2201_wy1 = -sign2 * layer2201_wv1 * np.sin(5. * np.pi / 180.) + layer2201_wu * np.cos(5. * np.pi / 180.)
    layer2201_wy2 = -sign2 * layer2201_wv2 * np.sin(5. * np.pi / 180.) + layer2201_wu * np.cos(5. * np.pi / 180.)
    layer2201_wz = np.ones(layer2201.shape[0])*z2201

    z2210 = z2200 + 2.6
    layer2210_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer2210.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2210_wv1 = -250. * np.ones(len(layer2210))
    layer2210_wv2 = 250. * np.ones(len(layer2210))
    layer2210_wx1 = layer2210_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer2210_wu * np.sin(5. * np.pi / 180.)
    layer2210_wx2 = layer2210_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer2210_wu * np.sin(5. * np.pi / 180.)
    layer2210_wy1 = -sign2 * layer2210_wv1 * np.sin(5. * np.pi / 180.) + layer2210_wu * np.cos(5. * np.pi / 180.)
    layer2210_wy2 = -sign2 * layer2210_wv2 * np.sin(5. * np.pi / 180.) + layer2210_wu * np.cos(5. * np.pi / 180.)
    layer2210_wz = np.ones(layer2210.shape[0])*z2210

    z2211 = z2210 + 1.1
    layer2211_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer2211.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer2211_wv1 = -250. * np.ones(len(layer2211))
    layer2211_wv2 = 250. * np.ones(len(layer2211))
    layer2211_wx1 = layer2211_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer2211_wu * np.sin(5. * np.pi / 180.)
    layer2211_wx2 = layer2211_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer2211_wu * np.sin(5. * np.pi / 180.)
    layer2211_wy1 = -sign2 * layer2211_wv1 * np.sin(5. * np.pi / 180.) + layer2211_wu * np.cos(5. * np.pi / 180.)
    layer2211_wy2 = -sign2 * layer2211_wv2 * np.sin(5. * np.pi / 180.) + layer2211_wu * np.cos(5. * np.pi / 180.)
    layer2211_wz = np.ones(layer2211.shape[0])*z2211
    
    layers = [layer1100, layer1101, layer1110, layer1111,
              layer1200, layer1201, layer1210, layer1211,
              layer2100, layer2101, layer2110, layer2111,
              layer2200, layer2201, layer2210, layer2211]

    layers_wu = [layer1100_wu, layer1101_wu, layer1110_wu, layer1111_wu,
              layer1200_wu, layer1201_wu, layer1210_wu, layer1211_wu,
              layer2100_wu, layer2101_wu, layer2110_wu, layer2111_wu,
              layer2200_wu, layer2201_wu, layer2210_wu, layer2211_wu]
    Wu = np.concatenate(layers_wu, axis=0)

    layers_wv1 = [layer1100_wv1, layer1101_wv1, layer1110_wv1, layer1111_wv1,
              layer1200_wv1, layer1201_wv1, layer1210_wv1, layer1211_wv1,
              layer2100_wv1, layer2101_wv1, layer2110_wv1, layer2111_wv1,
              layer2200_wv1, layer2201_wv1, layer2210_wv1, layer2211_wv1]
    Wv1 = np.concatenate(layers_wv1, axis=0)

    layers_wv2 = [layer1100_wv2, layer1101_wv2, layer1110_wv2, layer1111_wv2,
              layer1200_wv2, layer1201_wv2, layer1210_wv2, layer1211_wv2,
              layer2100_wv2, layer2101_wv2, layer2110_wv2, layer2111_wv2,
              layer2200_wv2, layer2201_wv2, layer2210_wv2, layer2211_wv2]
    Wv2 = np.concatenate(layers_wv2, axis=0)

    layers_wx1 = [layer1100_wx1, layer1101_wx1, layer1110_wx1, layer1111_wx1,
              layer1200_wx1, layer1201_wx1, layer1210_wx1, layer1211_wx1,
              layer2100_wx1, layer2101_wx1, layer2110_wx1, layer2111_wx1,
              layer2200_wx1, layer2201_wx1, layer2210_wx1, layer2211_wx1]
    Wx1 = np.concatenate(layers_wx1, axis=0)

    layers_wx2 = [layer1100_wx2, layer1101_wx2, layer1110_wx2, layer1111_wx2,
              layer1200_wx2, layer1201_wx2, layer1210_wx2, layer1211_wx2,
              layer2100_wx2, layer2101_wx2, layer2110_wx2, layer2111_wx2,
              layer2200_wx2, layer2201_wx2, layer2210_wx2, layer2211_wx2]
    Wx2 = np.concatenate(layers_wx2, axis=0)

    layers_wy1 = [layer1100_wy1, layer1101_wy1, layer1110_wy1, layer1111_wy1,
              layer1200_wy1, layer1201_wy1, layer1210_wy1, layer1211_wy1,
              layer2100_wy1, layer2101_wy1, layer2110_wy1, layer2111_wy1,
              layer2200_wy1, layer2201_wy1, layer2210_wy1, layer2211_wy1]
    Wy1 = np.concatenate(layers_wy1, axis=0)

    layers_wy2 = [layer1100_wy2, layer1101_wy2, layer1110_wy2, layer1111_wy2,
              layer1200_wy2, layer1201_wy2, layer1210_wy2, layer1211_wy2,
              layer2100_wy2, layer2101_wy2, layer2110_wy2, layer2111_wy2,
              layer2200_wy2, layer2201_wy2, layer2210_wy2, layer2211_wy2]
    Wy2 = np.concatenate(layers_wy2, axis=0)

    layers_wz = [layer1100_wz, layer1101_wz, layer1110_wz, layer1111_wz,
              layer1200_wz, layer1201_wz, layer1210_wz, layer1211_wz,
              layer2100_wz, layer2101_wz, layer2110_wz, layer2111_wz,
              layer2200_wz, layer2201_wz, layer2210_wz, layer2211_wz]
    Wz = np.concatenate(layers_wz, axis=0)

    data = pd.concat(layers, axis=0)
    data['Wu'] = Wu
    data['Wv1'] = Wv1
    data['Wv2'] = Wv2
    data['Wx1'] = Wx1
    data['Wx2'] = Wx2
    data['Wy1'] = Wy1
    data['Wy2'] = Wy2
    data['Wz'] = Wz

    zlayers = {0: z1100,
               1: z1101,
               2: z1110,
               3: z1111,
               4: z1200,
               5: z1201,
               6: z1210,
               7: z1211,
               8: z2100,
               9: z2101,
               10: z2110,
               11: z2111,
               12: z2200,
               13: z2201,
               14: z2210,
               15: z2211}

    return data, zlayers

def modify_for_xz_analysis_3_4(event):
    """
    Gets table of hits, fetchs only hits from U,V-views (3 & 4 stations) and adds to it columns 'Wz', 'Wu'- 
    coordinates of centres of tubes in plane (z, y).
    
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
    layer3100 = selector(event, 3, 1, 0, 0)
    layer3101 = selector(event, 3, 1, 0, 1)
    layer3110 = selector(event, 3, 1, 1, 0)
    layer3111 = selector(event, 3, 1, 1, 1)

    layer3200 = selector(event, 3, 2, 0, 0)
    layer3201 = selector(event, 3, 2, 0, 1)
    layer3210 = selector(event, 3, 2, 1, 0)
    layer3211 = selector(event, 3, 2, 1, 1)

    layer4100 = selector(event, 4, 1, 0, 0)
    layer4101 = selector(event, 4, 1, 0, 1)
    layer4110 = selector(event, 4, 1, 1, 0)
    layer4111 = selector(event, 4, 1, 1, 1)

    layer4200 = selector(event, 4, 2, 0, 0)
    layer4201 = selector(event, 4, 2, 0, 1)
    layer4210 = selector(event, 4, 2, 1, 0)
    layer4211 = selector(event, 4, 2, 1, 1)

    sign1 = -1.
    sign2 = +1.
    
    #3-u
    z3100 = 3338. - 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer3100_wu = -(0.5*0.9828 + 1.76*(layer3100.StrawNb.values-1)) + 499 + 0.44
    layer3100_wv1 = -250. * np.ones(len(layer3100))
    layer3100_wv2 = 250. * np.ones(len(layer3100))
    layer3100_wx1 = layer3100_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer3100_wu * np.sin(5. * np.pi / 180.)
    layer3100_wx2 = layer3100_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer3100_wu * np.sin(5. * np.pi / 180.)
    layer3100_wy1 = -sign1 * layer3100_wv1 * np.sin(5. * np.pi / 180.) + layer3100_wu * np.cos(5. * np.pi / 180.)
    layer3100_wy2 = -sign1 * layer3100_wv2 * np.sin(5. * np.pi / 180.) + layer3100_wu * np.cos(5. * np.pi / 180.)
    layer3100_wz = np.ones(layer3100.shape[0])*z3100

    z3101 = z3100 + 1.1
    layer3101_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer3101.StrawNb.values-1)) + 499 + 0.44
    layer3101_wv1 = -250. * np.ones(len(layer3101))
    layer3101_wv2 = 250. * np.ones(len(layer3101))
    layer3101_wx1 = layer3101_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer3101_wu * np.sin(5. * np.pi / 180.)
    layer3101_wx2 = layer3101_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer3101_wu * np.sin(5. * np.pi / 180.)
    layer3101_wy1 = -sign1 * layer3101_wv1 * np.sin(5. * np.pi / 180.) + layer3101_wu * np.cos(5. * np.pi / 180.)
    layer3101_wy2 = -sign1 * layer3101_wv2 * np.sin(5. * np.pi / 180.) + layer3101_wu * np.cos(5. * np.pi / 180.)
    layer3101_wz = np.ones(layer3101.shape[0])*z3101

    z3110 = z3100 + 2.6
    layer3110_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer3110.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3110_wv1 = -250. * np.ones(len(layer3110))
    layer3110_wv2 = 250. * np.ones(len(layer3110))
    layer3110_wx1 = layer3110_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer3110_wu * np.sin(5. * np.pi / 180.)
    layer3110_wx2 = layer3110_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer3110_wu * np.sin(5. * np.pi / 180.)
    layer3110_wy1 = -sign1 * layer3110_wv1 * np.sin(5. * np.pi / 180.) + layer3110_wu * np.cos(5. * np.pi / 180.)
    layer3110_wy2 = -sign1 * layer3110_wv2 * np.sin(5. * np.pi / 180.) + layer3110_wu * np.cos(5. * np.pi / 180.)
    layer3110_wz = np.ones(layer3110.shape[0])*z3110

    z3111 = z3110 + 1.1
    layer3111_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer3111.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3111_wv1 = -250. * np.ones(len(layer3111))
    layer3111_wv2 = 250. * np.ones(len(layer3111))
    layer3111_wx1 = layer3111_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer3111_wu * np.sin(5. * np.pi / 180.)
    layer3111_wx2 = layer3111_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer3111_wu * np.sin(5. * np.pi / 180.)
    layer3111_wy1 = -sign1 * layer3111_wv1 * np.sin(5. * np.pi / 180.) + layer3111_wu * np.cos(5. * np.pi / 180.)
    layer3111_wy2 = -sign1 * layer3111_wv2 * np.sin(5. * np.pi / 180.) + layer3111_wu * np.cos(5. * np.pi / 180.)
    layer3111_wz = np.ones(layer3111.shape[0])*z3111

    #3-v
    z3200 = 3338. + 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer3200_wu = -(0.5*0.9828 + 1.76*(layer3200.StrawNb.values-1)) + 499 + 0.44
    layer3200_wv1 = -250. * np.ones(len(layer3200))
    layer3200_wv2 = 250. * np.ones(len(layer3200))
    layer3200_wx1 = layer3200_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer3200_wu * np.sin(5. * np.pi / 180.)
    layer3200_wx2 = layer3200_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer3200_wu * np.sin(5. * np.pi / 180.)
    layer3200_wy1 = -sign2 * layer3200_wv1 * np.sin(5. * np.pi / 180.) + layer3200_wu * np.cos(5. * np.pi / 180.)
    layer3200_wy2 = -sign2 * layer3200_wv2 * np.sin(5. * np.pi / 180.) + layer3200_wu * np.cos(5. * np.pi / 180.)
    layer3200_wz = np.ones(layer3200.shape[0])*z3200
    
    z3201 = z3200 + 1.1
    layer3201_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer3201.StrawNb.values-1)) + 499 + 0.44
    layer3201_wv1 = -250. * np.ones(len(layer3201))
    layer3201_wv2 = 250. * np.ones(len(layer3201))
    layer3201_wx1 = layer3201_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer3201_wu * np.sin(5. * np.pi / 180.)
    layer3201_wx2 = layer3201_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer3201_wu * np.sin(5. * np.pi / 180.)
    layer3201_wy1 = -sign2 * layer3201_wv1 * np.sin(5. * np.pi / 180.) + layer3201_wu * np.cos(5. * np.pi / 180.)
    layer3201_wy2 = -sign2 * layer3201_wv2 * np.sin(5. * np.pi / 180.) + layer3201_wu * np.cos(5. * np.pi / 180.)
    layer3201_wz = np.ones(layer3201.shape[0])*z3201

    z3210 = z3200 + 2.6
    layer3210_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer3210.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3210_wv1 = -250. * np.ones(len(layer3210))
    layer3210_wv2 = 250. * np.ones(len(layer3210))
    layer3210_wx1 = layer3210_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer3210_wu * np.sin(5. * np.pi / 180.)
    layer3210_wx2 = layer3210_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer3210_wu * np.sin(5. * np.pi / 180.)
    layer3210_wy1 = -sign2 * layer3210_wv1 * np.sin(5. * np.pi / 180.) + layer3210_wu * np.cos(5. * np.pi / 180.)
    layer3210_wy2 = -sign2 * layer3210_wv2 * np.sin(5. * np.pi / 180.) + layer3210_wu * np.cos(5. * np.pi / 180.)
    layer3210_wz = np.ones(layer3210.shape[0])*z3210

    z3211 = z3210 + 1.1
    layer3211_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer3211.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer3211_wv1 = -250. * np.ones(len(layer3211))
    layer3211_wv2 = 250. * np.ones(len(layer3211))
    layer3211_wx1 = layer3211_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer3211_wu * np.sin(5. * np.pi / 180.)
    layer3211_wx2 = layer3211_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer3211_wu * np.sin(5. * np.pi / 180.)
    layer3211_wy1 = -sign2 * layer3211_wv1 * np.sin(5. * np.pi / 180.) + layer3211_wu * np.cos(5. * np.pi / 180.)
    layer3211_wy2 = -sign2 * layer3211_wv2 * np.sin(5. * np.pi / 180.) + layer3211_wu * np.cos(5. * np.pi / 180.)
    layer3211_wz = np.ones(layer3211.shape[0])*z3211

    #4-u
    z4100 = 3538. - 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer4100_wu = -(0.5*0.9828 + 1.76*(layer4100.StrawNb.values-1)) + 499 + 0.44
    layer4100_wv1 = -250. * np.ones(len(layer4100))
    layer4100_wv2 = 250. * np.ones(len(layer4100))
    layer4100_wx1 = layer4100_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer4100_wu * np.sin(5. * np.pi / 180.)
    layer4100_wx2 = layer4100_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer4100_wu * np.sin(5. * np.pi / 180.)
    layer4100_wy1 = -sign1 * layer4100_wv1 * np.sin(5. * np.pi / 180.) + layer4100_wu * np.cos(5. * np.pi / 180.)
    layer4100_wy2 = -sign1 * layer4100_wv2 * np.sin(5. * np.pi / 180.) + layer4100_wu * np.cos(5. * np.pi / 180.)
    layer4100_wz = np.ones(layer4100.shape[0])*z4100

    z4101 = z4100 + 1.1
    layer4101_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer4101.StrawNb.values-1)) + 499 + 0.44
    layer4101_wv1 = -250. * np.ones(len(layer4101))
    layer4101_wv2 = 250. * np.ones(len(layer4101))
    layer4101_wx1 = layer4101_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer4101_wu * np.sin(5. * np.pi / 180.)
    layer4101_wx2 = layer4101_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer4101_wu * np.sin(5. * np.pi / 180.)
    layer4101_wy1 = -sign1 * layer4101_wv1 * np.sin(5. * np.pi / 180.) + layer4101_wu * np.cos(5. * np.pi / 180.)
    layer4101_wy2 = -sign1 * layer4101_wv2 * np.sin(5. * np.pi / 180.) + layer4101_wu * np.cos(5. * np.pi / 180.)
    layer4101_wz = np.ones(layer4101.shape[0])*z4101

    z4110 = z4100 + 2.6
    layer4110_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer4110.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4110_wv1 = -250. * np.ones(len(layer4110))
    layer4110_wv2 = 250. * np.ones(len(layer4110))
    layer4110_wx1 = layer4110_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer4110_wu * np.sin(5. * np.pi / 180.)
    layer4110_wx2 = layer4110_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer4110_wu * np.sin(5. * np.pi / 180.)
    layer4110_wy1 = -sign1 * layer4110_wv1 * np.sin(5. * np.pi / 180.) + layer4110_wu * np.cos(5. * np.pi / 180.)
    layer4110_wy2 = -sign1 * layer4110_wv2 * np.sin(5. * np.pi / 180.) + layer4110_wu * np.cos(5. * np.pi / 180.)
    layer4110_wz = np.ones(layer4110.shape[0])*z4110

    z4111 = z4110 + 1.1
    layer4111_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer4111.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4111_wv1 = -250. * np.ones(len(layer4111))
    layer4111_wv2 = 250. * np.ones(len(layer4111))
    layer4111_wx1 = layer4111_wv1 * np.cos(5. * np.pi / 180.) + sign1 * layer4111_wu * np.sin(5. * np.pi / 180.)
    layer4111_wx2 = layer4111_wv2 * np.cos(5. * np.pi / 180.) + sign1 * layer4111_wu * np.sin(5. * np.pi / 180.)
    layer4111_wy1 = -sign1 * layer4111_wv1 * np.sin(5. * np.pi / 180.) + layer4111_wu * np.cos(5. * np.pi / 180.)
    layer4111_wy2 = -sign1 * layer4111_wv2 * np.sin(5. * np.pi / 180.) + layer4111_wu * np.cos(5. * np.pi / 180.)
    layer4111_wz = np.ones(layer4111.shape[0])*z4111

    #4-v
    z4200 = 3538. + 5. - 0.25*(2.6 - 1.1 - 0.9828) - 1.1 - 0.5*0.9828
    layer4200_wu = -(0.5*0.9828 + 1.76*(layer4200.StrawNb.values-1)) + 499 + 0.44
    layer4200_wv1 = -250. * np.ones(len(layer4200))
    layer4200_wv2 = 250. * np.ones(len(layer4200))
    layer4200_wx1 = layer4200_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer4200_wu * np.sin(5. * np.pi / 180.)
    layer4200_wx2 = layer4200_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer4200_wu * np.sin(5. * np.pi / 180.)
    layer4200_wy1 = -sign2 * layer4200_wv1 * np.sin(5. * np.pi / 180.) + layer4200_wu * np.cos(5. * np.pi / 180.)
    layer4200_wy2 = -sign2 * layer4200_wv2 * np.sin(5. * np.pi / 180.) + layer4200_wu * np.cos(5. * np.pi / 180.)
    layer4200_wz = np.ones(layer4200.shape[0])*z4200

    z4201 = z4200 + 1.1
    layer4201_wu = -(0.5*0.9828 - 0.88 + 1.76*(layer4201.StrawNb.values-1)) + 499 + 0.44
    layer4201_wv1 = -250. * np.ones(len(layer4201))
    layer4201_wv2 = 250. * np.ones(len(layer4201))
    layer4201_wx1 = layer4201_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer4201_wu * np.sin(5. * np.pi / 180.)
    layer4201_wx2 = layer4201_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer4201_wu * np.sin(5. * np.pi / 180.)
    layer4201_wy1 = -sign2 * layer4201_wv1 * np.sin(5. * np.pi / 180.) + layer4201_wu * np.cos(5. * np.pi / 180.)
    layer4201_wy2 = -sign2 * layer4201_wv2 * np.sin(5. * np.pi / 180.) + layer4201_wu * np.cos(5. * np.pi / 180.)
    layer4201_wz = np.ones(layer4201.shape[0])*z4201

    z4210 = z4200 + 2.6
    layer4210_wu = -(0.5*0.9828 - 0.44 + 1.76*(layer4210.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4210_wv1 = -250. * np.ones(len(layer4210))
    layer4210_wv2 = 250. * np.ones(len(layer4210))
    layer4210_wx1 = layer4210_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer4210_wu * np.sin(5. * np.pi / 180.)
    layer4210_wx2 = layer4210_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer4210_wu * np.sin(5. * np.pi / 180.)
    layer4210_wy1 = -sign2 * layer4210_wv1 * np.sin(5. * np.pi / 180.) + layer4210_wu * np.cos(5. * np.pi / 180.)
    layer4210_wy2 = -sign2 * layer4210_wv2 * np.sin(5. * np.pi / 180.) + layer4210_wu * np.cos(5. * np.pi / 180.)
    layer4210_wz = np.ones(layer4210.shape[0])*z4210

    z4211 = z4210 + 1.1
    layer4211_wu = -(0.5*0.9828 - 0.44 - 0.88 + 1.76*(layer4211.StrawNb.values-1)) + 499 + 0.44 - 0.88
    layer4211_wv1 = -250. * np.ones(len(layer4211))
    layer4211_wv2 = 250. * np.ones(len(layer4211))
    layer4211_wx1 = layer4211_wv1 * np.cos(5. * np.pi / 180.) + sign2 * layer4211_wu * np.sin(5. * np.pi / 180.)
    layer4211_wx2 = layer4211_wv2 * np.cos(5. * np.pi / 180.) + sign2 * layer4211_wu * np.sin(5. * np.pi / 180.)
    layer4211_wy1 = -sign2 * layer4211_wv1 * np.sin(5. * np.pi / 180.) + layer4211_wu * np.cos(5. * np.pi / 180.)
    layer4211_wy2 = -sign2 * layer4211_wv2 * np.sin(5. * np.pi / 180.) + layer4211_wu * np.cos(5. * np.pi / 180.)
    layer4211_wz = np.ones(layer4211.shape[0])*z4211
    
    layers = [layer3100, layer3101, layer3110, layer3111,
              layer3200, layer3201, layer3210, layer3211,
              layer4100, layer4101, layer4110, layer4111,
              layer4200, layer4201, layer4210, layer4211]

    layers_wu = [layer3100_wu, layer3101_wu, layer3110_wu, layer3111_wu,
              layer3200_wu, layer3201_wu, layer3210_wu, layer3211_wu,
              layer4100_wu, layer4101_wu, layer4110_wu, layer4111_wu,
              layer4200_wu, layer4201_wu, layer4210_wu, layer4211_wu]
    Wu = np.concatenate(layers_wu, axis=0)

    layers_wv1 = [layer3100_wv1, layer3101_wv1, layer3110_wv1, layer3111_wv1,
              layer3200_wv1, layer3201_wv1, layer3210_wv1, layer3211_wv1,
              layer4100_wv1, layer4101_wv1, layer4110_wv1, layer4111_wv1,
              layer4200_wv1, layer4201_wv1, layer4210_wv1, layer4211_wv1]
    Wv1 = np.concatenate(layers_wv1, axis=0)

    layers_wv2 = [layer3100_wv2, layer3101_wv2, layer3110_wv2, layer3111_wv2,
              layer3200_wv2, layer3201_wv2, layer3210_wv2, layer3211_wv2,
              layer4100_wv2, layer4101_wv2, layer4110_wv2, layer4111_wv2,
              layer4200_wv2, layer4201_wv2, layer4210_wv2, layer4211_wv2]
    Wv2 = np.concatenate(layers_wv2, axis=0)

    layers_wx1 = [layer3100_wx1, layer3101_wx1, layer3110_wx1, layer3111_wx1,
              layer3200_wx1, layer3201_wx1, layer3210_wx1, layer3211_wx1,
              layer4100_wx1, layer4101_wx1, layer4110_wx1, layer4111_wx1,
              layer4200_wx1, layer4201_wx1, layer4210_wx1, layer4211_wx1]
    Wx1 = np.concatenate(layers_wx1, axis=0)

    layers_wx2 = [layer3100_wx2, layer3101_wx2, layer3110_wx2, layer3111_wx2,
              layer3200_wx2, layer3201_wx2, layer3210_wx2, layer3211_wx2,
              layer4100_wx2, layer4101_wx2, layer4110_wx2, layer4111_wx2,
              layer4200_wx2, layer4201_wx2, layer4210_wx2, layer4211_wx2]
    Wx2 = np.concatenate(layers_wx2, axis=0)

    layers_wy1 = [layer3100_wy1, layer3101_wy1, layer3110_wy1, layer3111_wy1,
              layer3200_wy1, layer3201_wy1, layer3210_wy1, layer3211_wy1,
              layer4100_wy1, layer4101_wy1, layer4110_wy1, layer4111_wy1,
              layer4200_wy1, layer4201_wy1, layer4210_wy1, layer4211_wy1]
    Wy1 = np.concatenate(layers_wy1, axis=0)

    layers_wy2 = [layer3100_wy2, layer3101_wy2, layer3110_wy2, layer3111_wy2,
              layer3200_wy2, layer3201_wy2, layer3210_wy2, layer3211_wy2,
              layer4100_wy2, layer4101_wy2, layer4110_wy2, layer4111_wy2,
              layer4200_wy2, layer4201_wy2, layer4210_wy2, layer4211_wy2]
    Wy2 = np.concatenate(layers_wy2, axis=0)

    layers_wz = [layer3100_wz, layer3101_wz, layer3110_wz, layer3111_wz,
              layer3200_wz, layer3201_wz, layer3210_wz, layer3211_wz,
              layer4100_wz, layer4101_wz, layer4110_wz, layer4111_wz,
              layer4200_wz, layer4201_wz, layer4210_wz, layer4211_wz]
    Wz = np.concatenate(layers_wz, axis=0)

    data = pd.concat(layers, axis=0)
    data['Wu'] = Wu
    data['Wv1'] = Wv1
    data['Wv2'] = Wv2
    data['Wx1'] = Wx1
    data['Wx2'] = Wx2
    data['Wy1'] = Wy1
    data['Wy2'] = Wy2
    data['Wz'] = Wz

    zlayers = {0: z3100,
               1: z3101,
               2: z3110,
               3: z3111,
               4: z3200,
               5: z3201,
               6: z3210,
               7: z3211,
               8: z4100,
               9: z4101,
               10: z4110,
               11: z4111,
               12: z4200,
               13: z4201,
               14: z4210,
               15: z4211}

    return data, zlayers

def conventor_xz(event, indicator):
    """
    Gets pd.DataFrame() and transforms it into dictionary.
    
    Args:
        event: pd.DataFrame() that necesserily contains columns 'Wz', 'Wu', 'dist2Wire', 'ViewNb';
        indicator: 0 = 1 & 2 stations, 1 = 3 & 4 stations.
    Returns:
        dictionary: keys are values of 'Wz'; values are stuctures with fields(Wu, dist2Wire, angle, x, index, used).
    """

    if (indicator):

        event, zlayers = modify_for_xz_analysis_3_4(event)

    else:

        event, zlayers = modify_for_xz_analysis_1_2(event)


    dictionary = {}

    for i in event.index:

        params = parametresXZ(event.Wu[i], event.dist2Wire[i], (event.ViewNb[i] - 1.5) * 2 * np.pi / 36, 50000, False)
        dictionary.setdefault(event.Wz[i], {})[i] = params

    return dictionary, zlayers

def get_xz(plane_k, plane_b, event):

    event['Wy'] = plane_k * event.Wz + plane_b
    event['Wx'] = (event.Wx2 - event.Wx1) / (event.Wy2 - event.Wy1) * (event.Wy - event.Wy1) + event.Wx1

    intercect_hits = event # [np.abs(event.Wx.values) < 250]

    return intercect_hits

def points_crossing_line_xz(k, b, width, hits, intersecting_hits, n_min):
    """
    Counts the number of points which intercept line with parametres: plane_k, plane_b, plane_width.
    If the result more than n_min than makes linnear regression on this points.
    
    Args:
        plane_k, plane_b, plane_width: parametres of line in 2d space (z, x). It's a hyperplane in 3d space;
        hits: dictionary that contains all hits, key is z coordinate of centre of tube, value is another
            dictionary with key = index, value = structure with fields-Wu, dist2Wire, angle, used;
        n_min: minimal number of hits intercepting a track;
        intersecting_hits: dictionary containing subset of indexes of hits wich are intersected by the line in 2d-space (z, y).
    Returns:
        indicator, crossing_points, lin_regr;
        indicator: false- means line with this parametres doesn't cover a track,
            true- line covers a track;
        crossing_points: array of indexes of points that determine the track;
        lin_regr: parametres k, b of linnear regression on crossing points.
    """

    crossing_points = []
    X = [] # for linear regression
    Z = []
    R = []
    weights = []
    n = 0 # number of touched layers
    marks = {}


    for z in intersecting_hits:

        marks[z] = []
        indicator = False

        lower_x = k * z + b - 1. * width #/ np.cos(np.arctan(k))
        upper_x = k * z + b + 1. * width #/ np.cos(np.arctan(k))

        for j in intersecting_hits[z]:

            if ((hits[z][j].x < upper_x) & (hits[z][j].x > lower_x) & (not hits[z][j].used) & (not indicator)):

                crossing_points.append(j)
                Z.append(z)
                X.append(hits[z][j].x)
                R.append(hits[z][j].dist2Wire)
                weights.append(1 / (hits[z][j].dist2Wire)**(0.5))
                marks[z].append(j)
                indicator = True

        if indicator:
            n += 1


    if n < n_min:

        return 0, crossing_points, [0., 0.]

    else:

        lin_regr = np.polyfit(Z, X, 1, w=weights)

        for z in marks:

            for i in marks[z]:

                hits[z][i].used = True

        return 1, crossing_points, lin_regr

def loop_xz(event, tracks, linking_table, n_min, width, ind):
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

    #hits = conventor_xz(event, ind)

    new_linking_table = {}
    new_tracks = {}

    x_coordinates = {}
    tmp = {}
    new_trackID = 1


    if (ind):

        start_zs = [3331.2793, 3332.3793]
        end_zs = [3544.9793, 3543.8793]

    else:

        start_zs = [2591.2793, 2592.3793]
        end_zs = [2803.8793, 2804.9793]


    if (ind):

        event, zlayers = modify_for_xz_analysis_3_4(event)

    else:

        event, zlayers = modify_for_xz_analysis_1_2(event)



    for track_id in tracks:

        hits, zlayers = conventor_xz(event, ind) #!!!!!!!!!!
        layers = zlayers.keys()
        layers.sort()
        ndrop = len(layers) - n_min

        intersecting_hits = {}
        n = 0

        k = tracks[track_id][0]
        b = tracks[track_id][1]

        hits_xz = get_xz(k, b, event)

        # TODO: the loop optimization
        for z in hits:

            y = k * z + b

            for hit_index in hits[z]:

                x = hits_xz.loc[[hit_index]].Wx.values[0]

                if ((x > -300) & (x < 300) & (not hits[z][hit_index].used)):

                    hits[z][hit_index].x = x
                    tmp[hit_index] = x
                    intersecting_hits.setdefault(z, []).append(hit_index)
                    n += 1

        if (n >= n_min):

            for m in range(16, n_min - 1, -1):

                for idrop in range(ndrop):

                    if m > 16 - idrop:
                        continue

                    for drop_iter in range(idrop + 1):

                        start_ind = layers[0] + drop_iter
                        start_z = zlayers[start_ind]

                        end_ind = layers[-1] - (idrop - drop_iter)
                        end_z = zlayers[end_ind]

                        if intersecting_hits.has_key(start_z) and intersecting_hits.has_key(end_z):

                            for i in intersecting_hits[start_z]:

                                for j in intersecting_hits[end_z]:

                                    if ((not hits[start_z][i].used) & (not hits[end_z][j].used)):

                                        new_k, new_b = get_plane((hits[start_z][i].x, start_z), (hits[end_z][j].x, end_z))

                                        indicator, crossing_points, lin_regr = \
                                            points_crossing_line_xz(new_k, new_b, width, hits, intersecting_hits, m)

                                        if indicator == 1:

                                            new_tracks[track_id * 10000 + new_trackID] = lin_regr
                                            new_linking_table[track_id * 10000 + new_trackID] = crossing_points

                                            for k in crossing_points:
                                                x_coordinates[k] = tmp[k]

                                            new_trackID += 1


    return new_tracks, new_linking_table, x_coordinates