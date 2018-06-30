#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:04:39 2018

@author: saintlyvi
"""

import os
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import somoclu

import feather
import time
from datetime import date

from experiment.algorithms.cluster_metrics import mean_index_adequacy, davies_bouldin_score #, cluster_dispersion_index
from support import log_dir, cluster_dir, results_dir

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

def kmeans(X, range_n_clusters, preprocessing = None):
    """
    This function applies the MiniBatchKmeans algorithm from sklearn on inputs X for range_n_clusters.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    Returns cluster stats, cluster centroids and cluster labels.
    """
    
    cluster_centroids = {}
    cluster_stats = {'kmeans':{}}
    cluster_lbls = {}
    
    if preprocessing == None:
        X = np.array(X)
    elif preprocessing == 'normalize':
        X = normalize(X)

    cluster_centroids[0] = {}
    cluster_stats['kmeans'][0] = {} 
    cluster_lbls[0] = {}
    
    for n_clust in range_n_clusters:
        
        clusterer = MiniBatchKMeans(n_clusters=n_clust, random_state=10)
                    
        #2 Train clustering algorithm
        tic = time.time()        
        clusterer.fit(X)
        cluster_labels = clusterer.predict(X)
        toc = time.time()
        
         ## Calculate scores
        cluster_stats['kmeans'][0] = clusterStats(cluster_stats['kmeans'][0], n_clust, X, cluster_labels, 
                                     preprocessing = preprocessing, transform = None,
                                     tic = tic, toc = toc)        
        cluster_centroids[0][n_clust] = clusterer.cluster_centers_ 
        cluster_lbls[0][n_clust] = cluster_labels
#        print(progress(n_clust, cluster_stats[n_clust]))    
    
    return cluster_stats, cluster_centroids, cluster_lbls

def som(X, range_n_dim, preprocessing = None, transform=None, **kwargs):
    """
    This function applies the self organising maps algorithm from somoclu on inputs X over square maps of range_n_dim.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    If kmeans = True, the KMeans algorithm from sklearn is applied to the SOM and returns clusters
    kwargs can be n_clusters = range(start, end, interval) OR list()
    Returns cluster stats, cluster centroids and cluster labels.
    """
    
    cluster_centroids = {}
    cluster_stats = {'som':{}} 
    cluster_lbls = {}
    
    if preprocessing == None:
        X = np.array(X)
    elif preprocessing == 'normalize':
        X = normalize(X)

    for dim in range_n_dim:    
        
        cluster_centroids[dim] = {}
        cluster_stats['som'][dim] = {} 
        cluster_lbls[dim] = {}
   
        nrow = ncol = dim
        tic = time.time()

        #2 Train clustering algorithm
        som = somoclu.Somoclu(nrow, ncol, compactsupport=False, maptype='planar')
        som.train(X)
        toc = time.time()

        if transform == None:
            m = np.arange(0, nrow*ncol, 1).reshape(nrow, ncol) #create empty matrix the size of the SOM
            k = [m[som.bmus[i][1],som.bmus[i][0]] for i in range(0, len(som.bmus))] #get cluster of SOM node and assign to input vecors based on bmus
            c = pd.DataFrame(X).assign(cluster=k).groupby('cluster').mean()
                    
            ## Calculate scores
            n = 0
            cluster_stats['som'][dim] = clusterStats(cluster_stats['som'][dim], 
                         n, X, cluster_labels = k, preprocessing = preprocessing, 
                         transform = transform, tic = tic, toc = toc)
            cluster_centroids[dim][n] = np.array(c)
            cluster_lbls[dim][n] = k
    
        else:
            transform = 'kmeans_'
            if kwargs is None:
                n_clust = [10]
            else:
                for key, value in kwargs.items():
                    if key == 'n_clusters':
                        n_clust = value
            for n in n_clust:
                clusterer = KMeans(n_clusters=n, random_state=10)
                som.cluster(algorithm=clusterer)
                m = som.clusters
                k = [m[som.bmus[i][1],som.bmus[i][0]] for i in range(0, len(som.bmus))] #get cluster of SOM node and assign to input vecors based on bmus
                c = pd.DataFrame(X).assign(cluster=k).groupby('cluster').mean()
                    
                ## Calculate scores
                cluster_stats['som'][dim] = clusterStats(cluster_stats['som'][dim], 
                             n, X, cluster_labels = k, preprocessing = preprocessing, 
                             transform = transform + str(n), tic = tic, toc = toc)
                cluster_centroids[dim][n] = np.array(c)
                cluster_lbls[dim][n] = k
    
    return cluster_stats, cluster_centroids, cluster_lbls
        
def saveResults(experiment_name, cluster_stats, cluster_centroids):   
    
    centroid_results = pd.DataFrame()
    
    for level1_key, level1_values in cluster_stats.items():
#        if level1_key == 'kmeans':
#            reform = {(level1_key, level2_key): values
#                      for level1_key, level2_dict in cluster_stats.items()
#                      for level2_key, values in level2_dict.items()}
#            level_names = ['algorithm','n_clust']
#            
#            for k in level1_values.keys():
#                c = pd.DataFrame(cluster_centroids[k])
#                c['n_clust'] = k
#                c['cluster_size'] = pd.DataFrame.from_dict(level1_values[k]['cluster_size'])
#                centroid_results = centroid_results.append(c)
#
#        elif level1_key == 'som':            
        reform = {(level1_key, level2_key, level3_key): values
                  for level1_key, level2_dict in cluster_stats.items()
                  for level2_key, level3_dict in level2_dict.items()
                  for level3_key, values      in level3_dict.items()}
        level_names = ['algorithm','som_dim','n_clust']
        
        c = pd.DataFrame()
        for k, v in level1_values.items():
            for i, j in v.items():
                c = pd.DataFrame(cluster_centroids[k][i])
                c['som_dim'] = k
                c['n_clust'] = i
                c['cluster_size'] = j['cluster_size']
                centroid_results = centroid_results.append(c)
                      
    evals = pd.DataFrame(reform).T
    eval_results = evals.rename_axis(level_names).reset_index()
    eval_results.drop(labels='cluster_size', axis=1, inplace=True)
    eval_results[['dbi','mia','silhouette']] = eval_results[['dbi','mia','silhouette']].astype(float)
    eval_results['date'] = date.today().isoformat()
    eval_results['experiment'] = experiment_name
    
    centroid_results.reset_index(inplace=True)
    centroid_results.rename({'index':'k'}, axis=1, inplace=True)
    centroid_results['date'] = date.today().isoformat()
    centroid_results['experiment'] = experiment_name
    
    #3 Save Results
    os.makedirs(log_dir , exist_ok=True)    
    os.makedirs(cluster_dir , exist_ok=True)    
    os.makedirs(results_dir , exist_ok=True)

    eval_results.to_csv(os.path.join(results_dir, 'cluster_results.csv'), 
                        mode='a', index=False)
    centroid_results.to_csv(os.path.join(cluster_dir, 
                                         experiment_name + '_centroids.csv'), index=False)

    print('All results recorded')
    return eval_results

def saveLabels(X, cluster_lbls, top, eval_results, experiment_name):    

    #Save labels for 10 best clusters
    best_lbls = eval_results.nsmallest(columns=['dbi','mia'], n=top).nlargest(columns='silhouette', n=top)[['n_clust','som_dim']].reset_index(drop=True)
    labels = pd.DataFrame(cluster_lbls)
    lbls = pd.DataFrame([labels.loc[best_lbls.loc[r,'n_clust'],
                                    best_lbls.loc[r,'som_dim']] for r in range(len(best_lbls))]).T
    
    lbls['ProfileID'] = X.reset_index()['ProfileID']
    lbls['date'] = X.reset_index()['date']
    lbls.set_index(['ProfileID','date'], inplace=True)
    best_label_data = pd.concat([best_lbls.T, lbls])

    wpath = os.path.join(cluster_dir, date.today().isoformat() + experiment_name + '_labels.csv')
    feather.write_dataframe(best_label_data, wpath)    
    
    return print('All results logged')

#scores = eval_results.pivot_table(values=['dbi','mia','silhouette'],index='n_clust',columns='dim',aggfunc= lambda x:x)