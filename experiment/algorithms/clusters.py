#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:04:39 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

import time

from features.feature_ts import dailyProfiles, resampleProfiles
from experiment.algorithms.cluster_metrics import davies_bouldin, mean_index_adequacy, cluster_dispersion_index

def progress(n_clusters, year, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats[year]['t0']
    s = "%s | %s : " % (n_clusters, year)
    s += "total %(total_sample)s profiles clustered; " % stats
    s += "this batch %(n_sample)s profiles " % stats[year]
    s += "\nsilhouette: %(silhouette).3f " % stats[year]
    s += "\ndbi: %(dbi).3f " % stats[year]
    s += "\nmia: %(mia).3f " % stats[year]
    s += "\ncluster_sizes: [%s] " % ', '.join(map(str, stats[year]['cluster_sizes']))
    s += "in %.2fs (%s profiles/s) \n" % (duration, stats[year]['n_sample'] / duration)
    return s

year_range = [1994,1995]
range_n_clusters = range(8, 9, 1)

def kmeans(year_range, range_n_clusters, **kwargs):
    """
    This function applies the MiniBatchKmeans algorithm with a partial_fit over year_range for range_n_clusters.
    returns cluster_stats and cluster_centroids
    """
    
    cluster_centroids = {}
    cluster_stats = {}
    
    for n_clusters in range_n_clusters:
        
        cluster_centroids[n_clusters] = {}
        cluster_stats[n_clusters] = {}
        cluster_stats[n_clusters]['total_sample'] = 0    
        
        for y in range(year_range[0], year_range[1]+1):
            
            stats = {'n_sample': 0,
                 'cluster_sizes': [],
                 'silhouette': 0.0,
                 'dbi': 0.0,
                 'mia': 0.0,
    #             'cdi': 0.0,
                 't0': time.time(),
                 'batch_fit_time': 0.0}
            cluster_stats[n_clusters][y] = stats
            
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
        #TODO 1a data preprocessing - normalise data 
            
        #2 Clustering
            tick = time.time()
            
            # update estimator with examples in the current mini-batch
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
            clusterer.partial_fit(Xbatch)
            cluster_labels = clusterer.predict(Xbatch)
            
            try:
                ## Calculate scores
                cluster_centroids[n_clusters][y] = clusterer.cluster_centers_
                
                cluster_stats[n_clusters]['total_sample'] += Xbatch.shape[0]
                cluster_stats[n_clusters][y]['n_sample'] = Xbatch.shape[0]
                cluster_stats[n_clusters][y]['silhouette'] = silhouette_score(Xbatch, cluster_labels, sample_size=10000)
                cluster_stats[n_clusters][y]['dbi'] = davies_bouldin(Xbatch, cluster_labels)
                cluster_stats[n_clusters][y]['mia'] = mean_index_adequacy(Xbatch, cluster_labels)
        #        cluster_stats[n_clusters][y]['cdi'] =cluster_dispersion_index(Xbatch, cluster_labels) DON'T RUN LOCALLY!!
                cluster_stats[n_clusters][y]['cluster_sizes'] = np.bincount(cluster_labels)
                cluster_stats[n_clusters][y]['batch_fit_time'] += time.time() - tick
            except:
                pass
            
            print(progress(n_clusters, y, cluster_stats[n_clusters]))
    
    return cluster_stats, cluster_centroids
        
    #3 Plot Results

def kmeans_results(cluster_stats, cluster_centroids, year_range):   
    
    cluster_results = pd.DataFrame()
    eval_results = pd.DataFrame()
    
    for k in cluster_stats.keys():
        
        eval_metrics = pd.DataFrame.from_dict([cluster_stats[k][y][i] for i in ['n_sample','dbi','mia','silhouette','batch_fit_time']] for y in year_range)
        eval_metrics['year'] = year_range
        eval_metrics.columns = ['n_sample','dbi','mia','silhouette','batch_fit_time','year']
        eval_metrics['n_clusters'] = k
        eval_results = eval_results.append(eval_metrics)
        
        centroids = pd.DataFrame()
        for y in year_range:
            c = pd.DataFrame(cluster_centroids[k][y])
            c['year'] = y
            c['cluster_sizes'] = pd.DataFrame.from_dict(cluster_stats[k][y]['cluster_sizes'])
            centroids = centroids.append(c)
            
        centroids.reset_index(inplace=True)
        centroids.rename({'index':'k'}, axis=1, inplace=True)
        centroids['n_clusters'] = k
        
        cluster_results = cluster_results.append(centroids)

    cluster_results.reset_index(drop=True, inplace=True)
    cluster_results = cluster_results.set_index(['n_clusters','year','k'])
        
    return eval_results, cluster_results
    
    #4 Log Results
