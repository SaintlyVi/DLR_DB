#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 10:53:07 2018

@author: saintlyvi
"""

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances

#Average Squared-Loss Mutual Information Error (SMI),
#Violation rate of Root Squared Error (VRSE)
#Modified Dunn Index (MDI) 
#Cluster Dispersion Indicator (CDI)

def davies_bouldin(X, labels):
    """
    The DBI is the average of the similarity measures of each cluster with its most similar cluster. It captures cluster compactness and distinctness. The lower the score, the better.
    
    based on methods described in 10.1109/TPAMI.1979.4766909
    """    
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    
    # calculate cluster dispersion
    S = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []

    for i in range(n_cluster):
        Rij = []
        # establish similarity between each cluster and all other clusters
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / euclidean(centroids[i], centroids[j])
                Rij.append(r)
        # select Rij value of most similar cluster
        Ri.append(max(Rij)) 
    
    # get mean of all Ri values    
    dbi = np.mean(Ri)
    
    return dbi

def mean_index_adequacy(X, labels):
    """
    based on methods described in 10.3390/app8020237
    
    """
    
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    
    # calculate cluster dispersion
    D = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]

    # get mean of distance values    
    mia = np.sqrt(np.mean(D))
    
    return mia

def cluster_dispersion_index(X, labels):
    """
    based on methods described in 10.1109/TPWRS.2006.873122
    
    """
    
    n_clusters = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    
    Dclust = np.sqrt(np.sum(
                            [np.sqrt(np.sum(
                                    np.square(euclidean_distances(cluster_k[i]) ) )/(2*len(cluster_k[i]) )
                                    ) for i in range(n_clusters)]) / n_clusters )
    Dcent = np.sqrt(np.sum(np.square(euclidean_distances(centroids) ) )/(2*n_clusters) )
    
    cdi = Dclust/Dcent
    
    return cdi

