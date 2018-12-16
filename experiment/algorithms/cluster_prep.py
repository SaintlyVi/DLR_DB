#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:16:56 2018

@author: SaintlyVi
"""

import os
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import feather
import time
from datetime import date

from experiment.algorithms.cluster_metrics import mean_index_adequacy, davies_bouldin_score #, cluster_dispersion_index
from support import cluster_dir, results_dir
from features.feature_ts import resampleProfiles

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
         'all_scores': 0.0,
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
        cluster_stats[n]['all_scores'] = cluster_stats[n]['dbi']*cluster_stats[n]['mia']/cluster_stats[n]['silhouette']

        s = "%s : " % (n)                    
        s += "\nsilhouette: %(silhouette).3f " % stats
        s += "\ndbi: %(dbi).3f " % stats
        s += "\nmia: %(mia).3f " % stats
        print(s)

    except:
        print('Could not compute clustering stats for n = ' + str(n))
        pass

    return cluster_stats

def saveResults(experiment_name, cluster_stats, cluster_centroids, som_dim, elec_bin, save=True):
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
    evals['elec_bin'] = elec_bin
    eval_results = evals.drop(labels='cluster_size', axis=1).reset_index(drop=True)
#    eval_results.rename({'index':'k'}, axis=1, inplace=True)
    eval_results[['dbi','mia','silhouette']] = eval_results[['dbi','mia','silhouette']].astype(float)
    eval_results['date'] = date.today().isoformat()
#    eval_results['best_clusters'] = None

    centroid_results = pd.DataFrame(cluster_centroids)   
    centroid_results['experiment_name'] = experiment_name
    centroid_results['som_dim'] = som_dim
    centroid_results['n_clust'] = n
    centroid_results['elec_bin'] = elec_bin
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

def xBins(X, bin_type):

    if bin_type == 'amd':
        Xdd_A = X.sum(axis=1)
        Xdd = Xdd_A*230/1000
        XmonthlyPower = resampleProfiles(Xdd, interval='M', aggfunc='sum')
        Xamd = resampleProfiles(XmonthlyPower, interval='A', aggfunc='mean').reset_index().groupby('ProfileID').mean()
        Xamd.columns=['amd']
        
        amd_bins = [0, 1, 50, 150, 400, 600, 1200, 2500, 4000]    
        bin_labels = ['{0:.0f}-{1:.0f}'.format(x,y) for x, y in zip(amd_bins[:-1], amd_bins[1:])]    
        Xamd['bins'] = pd.cut(Xamd.amd, amd_bins, labels=bin_labels, right=True, include_lowest=True)
        
        Xbin_dict = dict()
        for c in Xamd.bins.cat.categories:
            Xbin_dict[c] = Xamd[Xamd.bins==c].index.values
        
        del Xdd_A, Xdd, XmonthlyPower, Xamd
        
    if bin_type == 'integral':
        Xint = normalize(X).cumsum(axis=1)
        Xintn = pd.DataFrame(Xint, index=X.index)
        Xintn['max'] = X.max(axis=1)
        clusterer = MiniBatchKMeans(n_clusters=8, random_state=10)
        clusterer.fit(np.array(Xintn))
        cluster_labels = clusterer.predict(np.array(Xintn))
        labl = pd.DataFrame(cluster_labels, index=X.index) 
        Xbin_dict = dict()
        for c in labl[0].unique():
            Xbin_dict['bin'+str(c)] = labl[labl[0]==c].index.values
            
    return Xbin_dict

def preprocessX(X, norm=None):  
    
    if norm == 'unit_norm': #Kwac et al 2013
        Xnorm = normalize(X)
    elif norm == 'zero-one': #Dent et al 2014
        Xnorm = np.array(X.divide(X.max(axis=1), axis=0))
    elif norm == 'demin': #Jin et al 2016
        Xnorm = normalize(X.subtract(X.min(axis=1), axis=0))
    elif norm == 'sa_norm': #Dekenah 2014
        Xnorm = np.array(X.divide(X.mean(axis=1), axis=0))
    else:
        Xnorm = np.array(X)
    
    #Xnorm.fillna(0, inplace=True)
    Xnorm[np.isnan(Xnorm)] = 0
        
    return Xnorm

def bestClusters(cluster_lbls, stats, top_lbls):

    labels = pd.DataFrame(cluster_lbls)
    
    if len(labels) > top_lbls:    
#        best_lbls = stats.nsmallest(columns=['dbi','mia'], n=top_lbls).nlargest(columns='silhouette',
#                                          n=top_lbls)[['n_clust','som_dim']].reset_index(drop=True)
#        b = stats.dbi*stats.mia/stats.silhouette
        stats.all_scores = stats.all_scores.astype('float')
        best_lbls = stats[stats.all_scores>0].nsmallest(columns='all_scores', n=top_lbls 
                         ).reset_index(drop=True)
        best_clusters = labels.loc[:, best_lbls['n_clust'].values]    
    
    else:
        best_lbls = stats[['n_clust','som_dim','elec_bin']]
        best_clusters = labels
    
#    best_clusters.columns = pd.MultiIndex.from_arrays([best_lbls['som_dim'], best_lbls['n_clust']],names=('som_dim','n_clust'))    
    stats.loc[stats['n_clust'].isin(best_lbls['n_clust'].values), 'best_clusters'] = 1
    
    return best_clusters, stats
       
def saveLabels(cluster_lbls, stats):    

    experiment_name = stats.experiment_name[0]
    elec_bin = stats.elec_bin[0]
    best_lbls = stats.loc[stats.best_clusters==1,['n_clust','som_dim','elec_bin']]
    best_lbls['experiment_name'] = experiment_name     
      
#    cluster_lbls[['ProfileID','date']] = pd.DataFrame(X).reset_index()[['ProfileID','date']]
#    cluster_lbls.set_index(['ProfileID','date'], inplace=True)
#    cluster_lbls.columns = pd.MultiIndex.from_arrays([best_lbls['som_dim'], best_lbls['n_clust']],names=('som_dim','n_clust'))
#    cluster_lbls.dropna(inplace=True)    
    cols = []
# TO DO this column order is wrong!!
    for i, j in zip(best_lbls['som_dim'],best_lbls['n_clust']):
        cols.append(str(i)+'_'+str(j))
    print(cols)
    cluster_lbls.columns = cols

    wpath = os.path.join(cluster_dir, experiment_name + '_' + elec_bin + '_labels.feather')
    feather.write_dataframe(cluster_lbls, wpath)
    
    blpath = os.path.join(results_dir, 'best_clusters.csv')
    if os.path.isfile(blpath):
        best_lbls.to_csv(blpath, mode='a', index=False, header=False)
    else:
        best_lbls.to_csv(blpath, index=False)
    
    return print('Labels for best '+experiment_name+' clusters saved')