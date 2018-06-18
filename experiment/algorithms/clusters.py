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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import somoclu

import time

from features.feature_ts import genX, dailyProfiles, resampleProfiles
from experiment.algorithms.cluster_metrics import davies_bouldin, mean_index_adequacy, cluster_dispersion_index, davies_bouldin_score
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
    
def clusterStats(cluster_stats, n, X, cluster_labels, preprocessing, transform, tic, toc):   
   
    stats = {'n_sample': 0,
         'cluster_size': [],
         'silhouette': 0.0,
         'dbi': 0.0,
         'mia': 0.0,
#             'cdi': 0.0,
         't0': time.time(),
         'batch_fit_time': 0.0,
         'total_sample': 0}

    cluster_stats[n] = stats
    try:
        cluster_stats[n]['total_sample'] += X.shape[0]
        cluster_stats[n]['n_sample'] = X.shape[0]
        cluster_stats[n]['silhouette'] = silhouette_score(X, cluster_labels, sample_size=10000)
        cluster_stats[n]['dbi'] = davies_bouldin_score(X, cluster_labels)
        cluster_stats[n]['mia'] = mean_index_adequacy(X, cluster_labels)
        #cluster_stats[n_clusters][y]['cdi'] =cluster_dispersion_index(Xbatch, cluster_labels) DON'T RUN LOCALLY!! - need to change to chunked alogrithm once released
        cluster_stats[n]['cluster_size'] = np.bincount(cluster_labels)
        cluster_stats[n]['batch_fit_time'] = toc - tic
        cluster_stats[n]['preprocessing'] = preprocessing
        cluster_stats[n]['transform'] = transform
    except:
        print('Could not compute clustering stats for n = ' + str(n))
        pass

    return cluster_stats

def kmeans(X, range_n_clusters, preprocessing = False, **kwargs):
    """
    This function applies the MiniBatchKmeans algorithm with a partial_fit over year_range for range_n_clusters.
    returns cluster_stats and cluster_centroids
    """
    
    cluster_centroids = {}
    cluster_stats = {}
    cluster_lbls = {}
    
    if preprocessing == False:
        X = np.array(X)
        preprocessing = None
    elif preprocessing == 'normalize':
        X = normalize(X)
    
    for n_clust in range_n_clusters:
        
        clusterer = MiniBatchKMeans(n_clusters=n_clust, random_state=10)
                    
        #2 Train clustering algorithm
        tic = time.time()        
        clusterer.fit(X)
        cluster_labels = clusterer.predict(X)
        toc = time.time()
        
         ## Calculate scores
        cluster_stats = clusterStats(cluster_stats, n_clust, X, cluster_labels, 
                                     preprocessing = preprocessing, transform = None,tic = tic,toc = toc)
        
        cluster_centroids[n_clust] = clusterer.cluster_centers_ 
        cluster_lbls[n_clust] = cluster_labels
#        print(progress(n_clust, cluster_stats[n_clust]))    
    
    return cluster_stats, cluster_centroids, cluster_lbls

def som(X, range_n_dim, preprocessing = False, algorithm=None):
    
    cluster_centroids = {}
    cluster_stats = {} 
    cluster_lbls = {}
    
    if preprocessing == False:
        X = np.array(X)
        preprocessing = None
    elif preprocessing == 'normalize':
        X = normalize(X)

    for dim in range_n_dim:        

        nrow = ncol = dim
        tic = time.time()
        #2 Train clustering algorithm
        som = somoclu.Somoclu(nrow, ncol, compactsupport=False, maptype='planar')
        som.train(X)
        toc = time.time()

        if algorithm is None:
            transform = None
            m = np.arange(0, nrow*ncol, 1).reshape(nrow, ncol)
            k = [m[som.bmus[i][1],som.bmus[i][0]] for i in range(0, len(som.bmus))]
            c = som.codebook.reshape(nrow * ncol, som.n_dim) #alternatively get mean of all k with same bmu

        else:
            transform = algorithm.get_params()
            som.cluster(algorithm=algorithm)
            k = [som.clusters[som.bmus[i][1],som.bmus[i][0]] for i in range(0, len(som.bmus))]
            c = #group codebook by k and get mean values or get mean of all Xi with same k
            
        ## Calculate scores
        cluster_stats = clusterStats(cluster_stats, dim, X, cluster_labels = k, 
                                     preprocessing = preprocessing, transform = transform, 
                                     tic = tic, toc = toc)

        cluster_centroids[dim] = c
        cluster_lbls[dim] = k
    
    return cluster_stats, cluster_centroids, cluster_lbls
        
def clusterResults(cluster_stats, cluster_centroids, cluster_lbls):   
    
    centroid_results = pd.DataFrame()
    eval_results = pd.DataFrame()
    
    eval_results = pd.DataFrame.from_dict([cluster_stats[k][i] for i in ['n_sample','dbi','mia','silhouette','batch_fit_time']] for k in cluster_stats.keys())
    eval_results.columns = ['n_sample','dbi','mia','silhouette','batch_fit_time']
    eval_results['n_clust'] = cluster_stats.keys()
    
    for k in cluster_stats.keys():
        c = pd.DataFrame(cluster_centroids[k])
        c['n_clust'] = k
        c['cluster_size'] = pd.DataFrame.from_dict(cluster_stats[k]['cluster_size'])
        centroid_results = centroid_results.append(c)
            
    centroid_results.reset_index(inplace=True)
    centroid_results.rename({'index':'k'}, axis=1, inplace=True)
    
    #Save labels for 10 best clusters
    best_lbls = eval_results.nsmallest(columns=['dbi','mia'], n= 10).nlargest(columns='silhouette', n=10)['n_clust'].values
#    best_lbls = [str(l) for l in best_lbls]
    
    labels = pd.DataFrame(cluster_lbls)
    lbls = labels.loc[:,best_lbls] 
    
    kfirst = list(cluster_stats.keys())[0]
    klast = list(cluster_stats.keys())[-1]
    it = int((klast-kfirst)/(len(list(cluster_stats.keys()))-1))
    #3 Log Results
    eval_results.to_csv(os.path.join(log_dir, 'kmeans' + str(kfirst)+'-'+str(klast)+'-'+str(it)+'_eval.csv'),index=False)
    centroid_results.to_csv(os.path.join(log_dir, 'kmeans' + str(kfirst)+'-'+str(klast)+'-'+str(it) +'_centroids.csv'),index=False)
    lbls.to_csv(os.path.join(log_dir, 'kmeans' + str(kfirst)+'-'+str(klast)+'-'+str(it)+'_labels.csv'),index=False)
        
    return eval_results, centroid_results
