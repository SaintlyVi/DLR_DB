#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:37:54 2018

@author: saintlyvi
"""

import pandas as pd
import numpy as np
import datetime as dt
from math import ceil, log
import feather
import os

import features.feature_ts as ts
import observations.obs_processing as op
from support import data_dir

def readResults():
    cluster_results = pd.read_csv('results/cluster_results.csv')
    cluster_results.drop_duplicates(subset=['dbi','mia','experiment_name'],keep='last',inplace=True)
    cluster_results = cluster_results[cluster_results.experiment_name != 'test']
    cluster_results['clusters'] = cluster_results['n_clust'].where(cluster_results['n_clust'] > 0, 
                   cluster_results['som_dim']**2)
    cluster_results['series'] = cluster_results['som_dim'].where(cluster_results['n_clust'] != 0, '')
    cluster_results['score'] = cluster_results.dbi*cluster_results.mia/cluster_results.silhouette
    
    return cluster_results

def bestClusters(cluster_results, n_best, experiment='all' ):
    if experiment=='all':
        experiment_clusters = cluster_results
    else:
        experiment_clusters = cluster_results.loc[cluster_results.experiment_name==experiment,:]
    best_clusters = experiment_clusters.loc[experiment_clusters.score>0,['experiment_name','som_dim','n_clust','dbi','mia','silhouette','score']].nsmallest(columns='score',n=n_best).reset_index(drop=True)
        
    return best_clusters

def bestLabels(experiment):
    data = feather.read_dataframe(os.path.join(data_dir, 'cluster_results', experiment+'_labels.feather'))
    X = ts.genX([1994,2014])
    data.columns = ['0_'+str(l) for l in data.max()+1]
    data[['ProfileID','date']] = pd.DataFrame(X).reset_index()[['ProfileID','date']]
    del X #clear memory
    data.date = data.date.astype(dt.date)#pd.to_datetime(data.date)
    data.ProfileID = data.ProfileID.astype('category')
    data.set_index(['ProfileID','date'], inplace=True)
    data.sort_index(level=['ProfileID','date'], inplace=True)
    
    return data

def clusterColNames(data):    
    data.columns = ['Cluster '+str(x+1) for x in data.columns]
    return data

def demandCorr(experiment, n_clust):
    data = feather.read_dataframe(os.path.join(data_dir, 'cluster_results', experiment+'_labels.feather'))
    X = ts.genX([1994,2014])
    Xdd = X.sum(axis=1).reset_index()
    Xdd.columns = ['ProfileID','date','DD_A']
    data[['ProfileID','date','DD_A']] = pd.DataFrame(Xdd)[['ProfileID','date','DD_A']]
    del X, Xdd #clear memory
    
    data.date = data.date.astype(dt.date)#pd.to_datetime(data.date)
    data.ProfileID = data.ProfileID.astype('category')
    
    
    return

def weekdayCorr(label_data, n_clust):
    df = label_data.reset_index()
    weekday_lbls = df.groupby([n_clust,df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    weekday_lbls = clusterColNames(weekday_lbls)
    weekday_likelihood = weekday_lbls.divide(weekday_lbls.sum(axis=0), axis=1) # likelihood of assignment
    
    random_likelihood = 1/len(weekday_likelihood) # null hypothesis
    relative_likelihood = weekday_likelihood.divide(random_likelihood, axis=1)

    random_loglike = log(random_likelihood,2)#-random_likelihood*log(random_likelihood,2)    
    weekday_loglike = weekday_likelihood.applymap(lambda x : log(x,2))#-x*log(x,2)) 
    relative_loglike = weekday_loglike.divide(random_loglike, axis=1)
    
    return weekday_likelihood, relative_likelihood

def monthlyCorr(label_data, n_clust):
    df = label_data.reset_index()
    month_lbls = df.groupby([n_clust,df.date.dt.month])['ProfileID'].count().unstack(level=0)
    month_lbls = clusterColNames(month_lbls)
    month_likelihood = month_lbls.divide(month_lbls.sum(axis=0), axis=1)
    
    random_likelihood = 1/len(month_likelihood)    
    relative_likelihood = month_likelihood.divide(random_likelihood, axis=1)    
    
    return month_likelihood, relative_likelihood

def yearlyCorr(label_data, n_clust):
    df = label_data.reset_index()
    year_lbls = df.groupby([n_clust,df.date.dt.year])['ProfileID'].count().unstack(level=0)
    year_lbls = clusterColNames(year_lbls).T
    year_likelihood = year_lbls.divide(year_lbls.sum(axis=0), axis=1)    
    
    random_likelihood = 1/len(year_likelihood)    
    relative_likelihood = year_likelihood.divide(random_likelihood, axis=1)
    
    return year_likelihood, relative_likelihood, 

def daytypeCorr(label_data, n_clust):
    df = label_data.reset_index()
    weekday_lbls = df.groupby([n_clust,df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    daytype = weekday_lbls[weekday_lbls.index.isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])].sum(axis=0).to_frame('weekday').T
    daytype = daytype.append(weekday_lbls.loc[['Saturday','Sunday'], :])
    daytype_lbls = clusterColNames(daytype)
    daytype_likelihood = daytype_lbls.divide(daytype_lbls.sum(axis=0), axis=1)

    random_likelihood = [5/7, 1/7, 1/7]
    relative_likelihood = daytype_likelihood.divide(random_likelihood, axis=0)
   
    return daytype_likelihood, relative_likelihood

def seasonCorr(label_data, n_clust):
    df = label_data.reset_index()
    month_lbls = df.groupby([n_clust,df.date.dt.month])['ProfileID'].count().unstack(level=0)    
    summer = month_lbls[~month_lbls.index.isin([5, 6, 7, 8])].sum(axis=0).to_frame('summer').T
    winter = month_lbls[month_lbls.index.isin([5, 6, 7, 8])].sum(axis=0).to_frame('winter').T        
    season = summer.append(winter)
    season_lbls = clusterColNames(season)
    season_likelihood = season_lbls.divide(season_lbls.sum(axis=0), axis=1)    
    
    random_likelihood = [8/12, 4/12]
    relative_likelihood = season_likelihood.divide(random_likelihood, axis=0)
    
    return season_likelihood, relative_likelihood

def clusterEntropy(likelihood, random_likelihood=None):
    if random_likelihood is None:
        try:
            random_likelihood = 1/len(likelihood)
        except:
            return('This function cannot compute entropy for weighted probabilities yet.')

    cluster_entropy = likelihood.applymap(lambda x : -x*log(x,2)).sum(axis=0)
    max_entropy = -random_likelihood*log(random_likelihood,2)*len(likelihood)
    
    ##TODO need to check how to calculate entropy when variables are weighted
    
    return cluster_entropy, max_entropy    
