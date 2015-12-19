import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from UV_views import *
from Y_views import *

def YZ_efficiency(all_hits, event_set, ind):
    """
    Shows efficiency of track recognition in Y_views. Efficiency is calculated as ratio of correctly 
    recognized hits to all hits.
    
    Args:
        all_hits: pd.Dataframe() containing all events;
        event_set: set of 'event' values to calculate the efficiency;
        ind: 0 = 1 & 2 stations, 1 = 3 & 4 stations.
    """
    match_table = {}
    for i in all_hits.Index:
        match_table[all_hits.Index[i]] = all_hits.TrackID[i]
    right_points = 0.
    all_points = 0.
    arr = []
    for event in event_set:
        event0 = all_hits[all_hits['event'] == event]
        if (ind):
            event0 = modify_for_yz_analysis_3_4(event0)
        else:
            event0 = modify_for_yz_analysis_1_2(event0)

        tracks, linking_table = loop_yz(event0, 7, 0.85, ind)
        for i in linking_table:
            per = {}
            for j in linking_table[i]:
                per[match_table[j]] = 0
            for j in linking_table[i]:
                per[match_table[j]] += 1
            if (len(linking_table[i]) > 0):
                arr.append(1.0 * max(per.values()) / len(linking_table[i]))
            right_points += max(per.values())
            all_points += len(linking_table[i])
    plt.hist(arr)
    print "efficiency:" + str(right_points / all_points)

def XZ_efficiency(all_hits, event_set, ind):
    """
    Shows efficiency of track recognition in UV_views. Efficiency is calculated as ratio of correctly 
    recognized hits to all hits.
    
    Args:
        all_hits: pd.Dataframe() containing all events;
        event_set: set of 'event' values to calculate the efficiency;
        ind: 0 = 1 & 2 stations, 1 = 3 & 4 stations.
    """
    match_table = {}
    for i in all_hits.Index:
        match_table[all_hits.Index[i]] = all_hits.TrackID[i]
    all_points = 0.
    right_points = 0.
    arr = []
    for event in event_set:
        event0 = all_hits[all_hits['event'] == event]
        if (ind):
            event1 = modify_for_yz_analysis_3_4(event0)
            event2 = modify_for_xz_analysis_3_4(event0)
        else:
            event1 = modify_for_yz_analysis_1_2(event0)
            event2 = modify_for_xz_analysis_1_2(event0)
        
        tracks, linking_table = loop_yz(event1, 7, 0.85, ind)
        new_tracks, new_linking_table, Xs = loop_xz(event2, tracks, linking_table, 6, 15, ind)
        for i in new_linking_table:
            per = {}
            for j in new_linking_table[i]:
                per[match_table[j]] = 0
            for j in new_linking_table[i]:
                per[match_table[j]] += 1
            if (len(new_linking_table[i]) > 0):
                arr.append(1.0 * max(per.values()) / len(new_linking_table[i]))
                right_points += max(per.values())
                all_points += len(new_linking_table[i])
    print "efficiency:" + str(right_points / all_points)
    plt.hist(arr)