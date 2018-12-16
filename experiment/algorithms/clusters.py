#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:04:39 2018

@author: saintlyvi
"""

import time
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans
import somoclu

from experiment.algorithms.cluster_prep import xBins, preprocessX, clusterStats, bestClusters, saveLabels, saveResults

def kmeans(X, range_n_clusters, top_lbls=10, preprocessing = None, bin_X=False, experiment_name=None):
    """
    This function applies the MiniBatchKmeans algorithm from sklearn on inputs X for range_n_clusters.
    If preprossing = True, X is normalised with sklearn.preprocessing.normalize()
    Returns cluster stats, cluster centroids and cluster labels.
    """

    if experiment_name is None:
        save = False
    else:
        if preprocessing is None:
            pass
        else:
            experiment_name = experiment_name+'_'+ preprocessing
        save = True
    
    #apply pre-binning
    if bin_X != False:
        Xbin = xBins(X, bin_X)
    else:
        Xbin = {'all':X}

    for b, ids in Xbin.items():
        try:
            A = X.loc[ids,:]
        except:
            A = ids

        #apply preprocessing    
        A = preprocessX(A, norm=preprocessing)

        centroids = pd.DataFrame()
        stats = pd.DataFrame() 
        cluster_lbls = pd.DataFrame()

        dim = 0 #set dim to 0 to match SOM formating  
        cluster_lbls_dim = {}
        stats_dim = pd.DataFrame()
        
        for n_clust in range_n_clusters:
            
            clusterer = MiniBatchKMeans(n_clusters=n_clust, random_state=10)
                        
            #train clustering algorithm
            tic = time.time()        
            clusterer.fit(A)
            cluster_labels = clusterer.predict(A)
            toc = time.time()
            
             ## Calculate scores
            cluster_stats = clusterStats({}, n_clust, A, cluster_labels, 
                                         preprocessing = preprocessing, transform = None,
                                         tic = tic, toc = toc)        
            cluster_centroids = clusterer.cluster_centers_ 
            
            eval_results, centroid_results = saveResults(experiment_name, cluster_stats,
                                                          cluster_centroids, dim, b, save)
            
            stats_dim = stats_dim.append(eval_results)
            centroids = centroids.append(centroid_results)
    
            cluster_lbls_dim[n_clust] = cluster_labels
    
        #outside n_clust loop
        best_clusters, best_stats = bestClusters(cluster_lbls_dim, stats_dim, top_lbls)
        cluster_lbls = pd.concat([cluster_lbls, best_clusters], axis=1)
        stats = pd.concat([stats, best_stats], axis=0)
        
        stats.reset_index(drop=True, inplace=True)

        if save is True:
            saveLabels(cluster_lbls, stats)
    
    return stats, centroids, cluster_lbls        

def som(X, range_n_dim, top_lbls=10, preprocessing = None, bin_X=False, transform=None, experiment_name=None, **kwargs):
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
        
    if experiment_name is None:
        save = False
    else:
        if preprocessing is None:
            pass
        else:
            experiment_name = experiment_name+'_'+ preprocessing
        save = True

    #apply pre-binning
    if bin_X != False:
        Xbin = xBins(X, bin_X)
    else:
        Xbin = {'0-4000':X}

    for b, ids in Xbin.items():
        try:
            A = X.loc[ids,:]
        except:
            A = ids
        #apply preprocessing    
        A = preprocessX(A, norm=preprocessing)

        centroids = pd.DataFrame()
        stats = pd.DataFrame() 
        cluster_lbls = pd.DataFrame()

        for dim in range_n_dim: 
            
            cluster_lbls_dim = {}
            stats_dim = pd.DataFrame()        
            nrow = ncol = dim
            tic = time.time()
    
            #train clustering algorithm
            som = somoclu.Somoclu(nrow, ncol, compactsupport=False, maptype='planar')
            som.train(A)
            toc = time.time()
    
            if transform == None:
                n_clust = [0]    
            elif transform == 'kmeans':
                if kwargs is None:
                    n_clust = [10]
                else:
                    for key, value in kwargs.items(): #create list with number of clusters for kmeans
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
                c = pd.DataFrame(A).assign(cluster=k).groupby('cluster').mean()
                    
                #calculate scores
                cluster_stats = clusterStats({}, n, A, cluster_labels = k, preprocessing = preprocessing,
                                             transform = transform, tic = tic, toc = toc)
                cluster_centroids = np.array(c)
                
                eval_results, centroid_results = saveResults(experiment_name, cluster_stats,
                                                             cluster_centroids, dim, b, save)
    
                stats_dim = stats_dim.append(eval_results)
                centroids = centroids.append(centroid_results)
    
                cluster_lbls_dim[n] = k
            
            #outside n_clust loop
            best_clusters, best_stats = bestClusters(cluster_lbls_dim, stats_dim, top_lbls)
            cluster_lbls = pd.concat([cluster_lbls, best_clusters],axis=1)
            stats = pd.concat([stats, best_stats], axis=0)
            
        stats.reset_index(drop=True, inplace=True)
        if save is True:
            saveLabels(cluster_lbls, stats)
    
    return stats, centroids, cluster_lbls