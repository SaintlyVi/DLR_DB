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
from support import cluster_dir, results_dir

def progress(n, stats):
    """Report progress information, return a string."""
    s = "%s : " % (n)                    
    s += "\nsilhouette: %(silhouette).3f " % stats
    s += "\ndbi: %(dbi).3f " % stats
    s += "\nmia: %(mia).3f " % stats  
    return print(s)
    
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

        s = "%s : " % (n)                    
        s += "\nsilhouette: %(silhouette).3f " % stats
        s += "\ndbi: %(dbi).3f " % stats
        s += "\nmia: %(mia).3f " % stats
        print(s)

    except:
        print('Could not compute clustering stats for n = ' + str(n))
        pass

    return cluster_stats

def saveResults(experiment_name, cluster_stats, cluster_centroids, som_dim, save=True):
    """
    Saves cluster stats results and centroids for a single clustering iteration. 
    Called inside kmeans() and som() functions.
    """

    for k, v in cluster_stats.items():
        n = k
                        
    evals = pd.DataFrame(cluster_stats).T
    evals['experiment_name'] = experiment_name
    evals['som_dim'] = som_dim
    evals['n_clust'] = n
    eval_results = evals.drop(labels='cluster_size', axis=1).reset_index(drop=True)
#    eval_results.rename({'index':'k'}, axis=1, inplace=True)
    eval_results[['dbi','mia','silhouette']] = eval_results[['dbi','mia','silhouette']].astype(float)
    eval_results['date'] = date.today().isoformat()
    eval_results['best_clusters'] = None

    centroid_results = pd.DataFrame(cluster_centroids)   
    centroid_results['experiment_name'] = experiment_name
    centroid_results['som_dim'] = som_dim
    centroid_results['n_clust'] = n
    try:
        centroid_results['cluster_size'] = evals['cluster_size'][n]
    except:
        centroid_results['cluster_size'] = np.nan
    centroid_results.reset_index(inplace=True)
    centroid_results.rename({'index':'k'}, axis=1, inplace=True)
    centroid_results['date'] = date.today().isoformat()
    
    #3 Save Results
    if save is True:
        os.makedirs(results_dir, exist_ok=True)    
        erpath = os.path.join(results_dir, 'cluster_results.csv')    
        if os.path.isfile(erpath):
            eval_results.to_csv(erpath, mode='a', index=False, header=False)
        else:
            eval_results.to_csv(erpath, index=False)

        os.makedirs(cluster_dir, exist_ok=True)   
        crpath = os.path.join(cluster_dir, experiment_name + '_centroids.csv')    
        if os.path.isfile(crpath):
            centroid_results.to_csv(crpath, mode='a', index=False, header=False)
        else:
            centroid_results.to_csv(crpath, index=False)
        
        print('Results saved for', experiment_name, str(som_dim), str(n))
    
    return eval_results, centroid_results

def kmeans(X, range_n_clusters, preprocessing = None, experiment_name=None):
    """
    This function applies the MiniBatchKmeans algorithm from sklearn on inputs X for range_n_clusters.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    Returns cluster stats, cluster centroids and cluster labels.
    """
    
    centroids = pd.DataFrame()
    stats = pd.DataFrame() 
    cluster_lbls = {}
    dim = 0 #set dim to 0 to match SOM formating
    cluster_lbls[dim] = {}
        
    if preprocessing == None:
        X = np.array(X)
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
        cluster_stats = clusterStats({}, n_clust, X, cluster_labels, 
                                     preprocessing = preprocessing, transform = None,
                                     tic = tic, toc = toc)        
        cluster_centroids = clusterer.cluster_centers_ 
        
        if experiment_name is None:
            save = False
        else:
            save = True
        eval_results, centroid_results = saveResults(experiment_name, cluster_stats,
                                                      cluster_centroids, dim, save)

        stats = stats.append(eval_results)
        centroids = centroids.append(centroid_results)

        cluster_lbls[dim][n_clust] = cluster_labels
    
    stats.reset_index(drop=True, inplace=True)
    
    return stats, centroids, cluster_lbls        

def som(X, range_n_dim, top_lbls=10, preprocessing = None, transform=None, experiment_name=None, **kwargs):
    """
    This function applies the self organising maps algorithm from somoclu on inputs X over square maps of range_n_dim.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    If kmeans = True, the KMeans algorithm from sklearn is applied to the SOM and returns clusters
    kwargs can be n_clusters = range(start, end, interval) OR list()
    Returns cluster stats, cluster centroids and cluster labels.
    """

    for dim in range_n_dim: 
        limit = int(np.sqrt(len(X)/20))
        if dim > limit: #verify that number of nodes are sensible for size of input data
            return print('Input size too small for map. Largest n should be ' + str(limit))
        else:
            pass
    
    centroids = pd.DataFrame()
    stats = pd.DataFrame() 
    cluster_lbls = pd.DataFrame()
    best_lbls = pd.DataFrame()
    
    if preprocessing == None:
        X = np.array(X)
    elif preprocessing == 'normalize':
        X = normalize(X)

    for dim in range_n_dim: 
        
        cluster_lbls_dim = {}        
        nrow = ncol = dim
        tic = time.time()

        #2 Train clustering algorithm
        som = somoclu.Somoclu(nrow, ncol, compactsupport=False, maptype='planar')
        som.train(X)
        toc = time.time()

        if transform == None:
            n_clust = [0]    
        elif transform == 'kmeans':
            if kwargs is None:
                n_clust = [10]
            else:
                for key, value in kwargs.items():
                    if key == 'n_clusters':
                        n_clust = value
        else:
            return('Cannot process this transform algorithm')
            
        for n in n_clust:
            if n == 0:
                #create empty matrix the size of the SOM
                m = np.arange(0, nrow*ncol, 1).reshape(nrow, ncol) 
            else:
                clusterer = KMeans(n_clusters=n, random_state=10)
                som.cluster(algorithm=clusterer)
                m = som.clusters
            #get cluster of SOM node and assign to input vecors based on bmus
            k = [m[som.bmus[i][1],som.bmus[i][0]] for i in range(0, len(som.bmus))] 
            c = pd.DataFrame(X).assign(cluster=k).groupby('cluster').mean()
                
            #calculate scores
            cluster_stats = clusterStats({}, n, X, cluster_labels = k, preprocessing = preprocessing, 
                         transform = transform, tic = tic, toc = toc)
            cluster_centroids = np.array(c)
            
            if experiment_name is None:
                save = False
            else:
                save = True
            eval_results, centroid_results = saveResults(experiment_name, cluster_stats,
                                                          cluster_centroids, dim, save)

            stats = stats.append(eval_results)
            centroids = centroids.append(centroid_results)

            cluster_lbls_dim[n] = k
        
        #outside n_clust loop
        lbls_dim, best_lbls_dim = bestClusters(cluster_lbls_dim, stats, top_lbls)
        cluster_lbls = pd.concat([cluster_lbls, lbls_dim],axis=1)
#        best_lbls = pd.concat([best_lbls, best_lbls_dim], axis=0)
        
    stats.reset_index(drop=True, inplace=True)
    
    return stats, centroids, cluster_lbls

def bestClusters(cluster_lbls, stats, top_lbls):

    labels = pd.DataFrame(cluster_lbls)
    
    if len(labels) > top_lbls:
    
        best_lbls = stats.nsmallest(columns=['dbi','mia'], n=top_lbls).nlargest(columns='silhouette',
                                          n=top_lbls)[['n_clust','som_dim']].reset_index(drop=True)
        lbls = labels.loc[:, best_lbls['n_clust'].values]    
    
    else:
        best_lbls = stats[['n_clust','som_dim']]
        lbls = labels
        
    best_lbls['date'] = date.today().isoformat()
    
    return lbls, best_lbls
       
def saveLabels(X, cluster_lbls, stats, top_lbls):    

    experiment_name = stats.experiment_name[0]    
    
    best_lbls = stats.nsmallest(columns=['dbi','mia'], n=top_lbls).nlargest(columns='silhouette',
                                      n=top_lbls)[['n_clust','som_dim']].reset_index(drop=True)
    best_lbls['experiment'] = experiment_name
    best_lbls['date'] = date.today().isoformat()

    os.makedirs(cluster_dir, exist_ok=True)
    blpath = os.path.join(cluster_dir, 'best_labels.csv')    
    if os.path.isfile(blpath):
        best_lbls.to_csv(os.path.join(blpath), mode='a', index=False, header=False)
    else:
        best_lbls.to_csv(os.path.join(blpath), index=False)

    #Save labels for 10 best clusters
    labels = pd.DataFrame(cluster_lbls)
    lbls = pd.DataFrame([labels.loc[best_lbls.loc[r,'n_clust'],
                                    best_lbls.loc[r,'som_dim']] for r in range(len(best_lbls))]).T
    
    lbls['ProfileID'] = X.reset_index()['ProfileID']
    lbls['date'] = X.reset_index()['date']
    lbls.set_index(['ProfileID','date'], inplace=True)
    lbls.dropna(inplace=True)

    wpath = os.path.join(cluster_dir, experiment_name + '_labels.feather')
    feather.write_dataframe(lbls, wpath)    
    
    return print('Labels for top '+str(top_lbls)+' clusters saved')

#scores = eval_results.pivot_table(values=['dbi','mia','silhouette'],index='n_clust',columns='dim',aggfunc= lambda x:x)