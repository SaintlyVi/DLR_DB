#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:04:39 2018

@author: saintlyvi
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

import time

from features.feature_ts import dailyProfiles, resampleProfiles
from experiment.algorithms.cluster_metrics import davies_bouldin, mean_index_adequacy, cluster_dispersion_index
from support import log_dir

def progress(n_clusters, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%s : " % (n_clusters)
    s += "total %(total_sample)s profiles clustered; " % stats
    s += "this batch %(n_sample)s profiles " % stats
    s += "\nsilhouette: %(silhouette).3f " % stats
    s += "\ndbi: %(dbi).3f " % stats
    s += "\nmia: %(mia).3f " % stats
    s += "\ncluster_sizes: [%s] " % ', '.join(map(str, stats['cluster_size']))
    s += "in %.2fs (%s profiles/s) \n" % (duration, stats['n_sample'] / duration)
    return s

def genX(year_range, **kwargs):
    
    X = pd.DataFrame()
    
    for y in range(year_range[0], year_range[1]+1):
               
    #1 Get minibatch
        if 'interval' in kwargs: interval = kwargs['interval']
        else: interval = None
            
        if 'aggfunc' in kwargs: aggfunc = kwargs['aggfunc']
        else: aggfunc = 'mean'
            
        if 'unit' in kwargs: unit = kwargs['unit']
        else: unit = 'A'
            
        if 'directory' in kwargs: directory = kwargs['directory']
        else: directory = 'H'
                 
        data = resampleProfiles(dailyProfiles(y, unit, directory), interval, aggfunc)
        Xbatch = data.dropna()
        Xbatch.reset_index(inplace=True)
        X = X.append(Xbatch)
    X.set_index(['ProfileID','date'],inplace=True)
    
    return X
    
#TODO 1a data preprocessing - normalise data 

def kmeans(X, range_n_clusters, normalise = False, **kwargs):
    """
    This function applies the MiniBatchKmeans algorithm with a partial_fit over year_range for range_n_clusters.
    returns cluster_stats and cluster_centroids
    """
    
    cluster_centroids = {}
    cluster_stats = {}  
    
    if normalise == False:
        pass
#    else:
#        X = normalize(X)
    
    for n_clust in range_n_clusters:
        
        cluster_centroids[n_clust] = {}
        
        stats = {'n_sample': 0,
             'cluster_size': [],
             'silhouette': 0.0,
             'dbi': 0.0,
             'mia': 0.0,
#             'cdi': 0.0,
             't0': time.time(),
             'batch_fit_time': 0.0,
             'total_sample': 0}
        cluster_stats[n_clust] = stats
        
        clusterer = MiniBatchKMeans(n_clusters=n_clust, random_state=10)
                    
        #2 Clustering
        tick = time.time()        
        # update estimator with examples in the current mini-batch
        clusterer.fit(X)
        cluster_labels = clusterer.predict(X)
            
        try:
            ## Calculate scores
            cluster_centroids[n_clust] = clusterer.cluster_centers_          
            cluster_stats[n_clust]['total_sample'] += X.shape[0]
            cluster_stats[n_clust]['n_sample'] = X.shape[0]
            cluster_stats[n_clust]['silhouette'] = silhouette_score(X, cluster_labels, sample_size=10000)
            cluster_stats[n_clust]['dbi'] = davies_bouldin(X, cluster_labels)
            cluster_stats[n_clust]['mia'] = mean_index_adequacy(X, cluster_labels)
    #        cluster_stats[n_clusters][y]['cdi'] =cluster_dispersion_index(Xbatch, cluster_labels) DON'T RUN LOCALLY!!
            cluster_stats[n_clust]['cluster_size'] = np.bincount(cluster_labels)
            cluster_stats[n_clust]['batch_fit_time'] += time.time() - tick
        except:
            pass
            
        print(progress(n_clust, cluster_stats[n_clust]))
    
    return cluster_stats, cluster_centroids
        
def kmeansResults(cluster_stats, cluster_centroids):   
    
    cluster_results = pd.DataFrame()
    eval_results = pd.DataFrame()
    
    eval_results = pd.DataFrame.from_dict([cluster_stats[k][i] for i in ['n_sample','dbi','mia','silhouette','batch_fit_time']] for k in cluster_stats.keys())
    eval_results.columns = ['n_sample','dbi','mia','silhouette','batch_fit_time']
    eval_results['n_clust'] = cluster_stats.keys()
    
    for k in cluster_stats.keys():
        c = pd.DataFrame(cluster_centroids[k])
        c['n_clust'] = k
        c['cluster_size'] = pd.DataFrame.from_dict(cluster_stats[k]['cluster_size'])
        cluster_results = cluster_results.append(c)
            
    cluster_results.reset_index(inplace=True)
    cluster_results.rename({'index':'k'}, axis=1, inplace=True)
    
    kfirst = list(cluster_stats.keys())[0]
    klast = list(cluster_stats.keys())[-1]
    it = int((klast-kfirst)/(len(list(cluster_stats.keys()))-1))
    #3 Log Results
    eval_results.to_csv(os.path.join(log_dir, str(kfirst)+'-'+str(klast)+'-'+str(it)+'_eval_kmeans.csv'),index=False)
    cluster_results.to_csv(os.path.join(log_dir, str(kfirst)+'-'+str(klast)+'-'+str(it) +'_cluster_kmeans.csv'),index=False)
        
    return eval_results, cluster_results
    
    #4 Plot Results
