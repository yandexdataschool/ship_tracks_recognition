import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from UV_views import *
from Y_views import *

def efficiency(recognized_tracks, match_table):
    """
    Calculates efficiency of recognized tracks.
    
    Args:
        recognized_tracks: dictionary of recognized tracks;
        match_table: dictionary(key = index of hit from data; value = TrackID).
        
    Returns:
        arr: array, elements of this array store percentages of rihght recognized hits for each track;
        eff: common efficiency.
    """
    arr = []
    right_points = 0.
    all_points = 0.
    for i in recognized_tracks:
        per = {}
        for j in recognized_tracks[i]:
            per[match_table[j]] = 0
        for j in recognized_tracks[i]:
            per[match_table[j]] += 1
        arr.append(1.0 * max(per.values()) / len(recognized_tracks[i]))
        right_points += max(per.values())
        all_points += len(recognized_tracks[i])
    return arr, right_points / all_points