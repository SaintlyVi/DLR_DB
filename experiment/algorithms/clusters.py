#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:04:39 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np
from sklearn.clusters import KMeans
from features.feature_ts import dailyProfiles

def bench_kmeans(year_start, year_end, unit, directory):

    p = pd.DataFrame()
    for y in range(year_start, year_end+1):
        data = dailyProfiles(y, unit, directory)
        p = p.append(data)
        
    p_prep = p.dropna()
    kmeans = KMeans(30, random_state=0).fit(p_prep)
    
    pc = p_prep.assign(cluster=kmeans.labels_)
    c = pc.reset_index(drop=True)
    c_centres = pd.DataFrame(kmeans.cluster_centers_)
    c_mean = c.groupby('cluster').mean()
    c_count = c.groupby('cluster').count()
    
    rmse = np.sqrt((c_centres - c_mean)**2).mean(axis=1) #RMSE between cluster centre values and cluster mean values

##TODO try different cluster numbers and calculate DBI for each. Select accordingly
    
    return kmeans, rmse