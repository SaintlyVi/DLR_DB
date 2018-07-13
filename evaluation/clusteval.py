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

def weekday_corr(label_data, n_clust):
    df = label_data.reset_index()
    weekday_lbls = df.groupby([n_clust,df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    weekday_lbls = clusterColNames(weekday_lbls)
    weekday_likelihood = weekday_lbls.divide(weekday_lbls.sum(axis=0), axis=1)    
    
    random_likelihood = 1/len(weekday_likelihood)  
#    weekday_ent = weekday_likelihood.divide(random_likelihood, axis=1)
    
    weekday_entropy = -weekday_likelihood.applymap(lambda x : x*log(x,2))
    
    return random_likelihood, weekday_entropy

def month_corr(label_data, n_clust):
    df = label_data.reset_index()
    month_lbls = df.groupby([n_clust,df.date.dt.month])['ProfileID'].count().unstack(level=0)
    month_lbls = clusterColNames(month_lbls)
    month_lbls2 = month_lbls.divide(month_lbls.sum(axis=0), axis=1)
    
    random_likelihood = 1/len(month_lbls2)
    
    month_lbls2 = month_lbls2.divide(random_likelihood, axis=1)    
    
    return random_likelihood, month_lbls2

def year_corr(label_data, n_clust):
    df = label_data.reset_index()
    year_lbls = df.groupby([n_clust,df.date.dt.year])['ProfileID'].count().unstack(level=0)
    year_lbls = clusterColNames(year_lbls).T
    year_lbls2 = year_lbls.divide(year_lbls.sum(axis=0), axis=1)    
    
    random_likelihood = 1/len(year_lbls2)
    
    year_lbls2 = year_lbls2.divide(random_likelihood, axis=1)
    
    return random_likelihood, year_lbls2

def daytype_corr(label_data, n_clust):
    df = label_data.reset_index()
    weekday_lbls = df.groupby([n_clust,df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    daytype = weekday_lbls[weekday_lbls.index.isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])].sum(axis=0).to_frame('weekday').T
    daytype = daytype.append(weekday_lbls.loc[['Saturday','Sunday'], :])
    daytype_lbls = clusterColNames(daytype)
    daytype_lbls2 = daytype_lbls.divide(daytype_lbls.sum(axis=0), axis=1)
    
    random_likelihood = 1/len(weekday_lbls)
    
    daytype_lbls2.loc['weekday'] = daytype_lbls2.loc['weekday'].apply(lambda x: x/(5*random_likelihood))
    daytype_lbls2.loc['Saturday'] = daytype_lbls2.loc['Saturday'].apply(lambda x: x/(random_likelihood))
    daytype_lbls2.loc['Sunday'] = daytype_lbls2.loc['Sunday'].apply(lambda x: x/(random_likelihood))    
    
    return random_likelihood, daytype_lbls2

def season_corr(label_data, n_clust):
    df = label_data.reset_index()
    month_lbls = df.groupby([n_clust,df.date.dt.month])['ProfileID'].count().unstack(level=0)    
    summer = month_lbls[~month_lbls.index.isin([5, 6, 7, 8])].sum(axis=0).to_frame('summer').T
    winter = month_lbls[month_lbls.index.isin([5, 6, 7, 8])].sum(axis=0).to_frame('winter').T        
    season = summer.append(winter)
    season_lbls = clusterColNames(season)
    season_lbls2 = season_lbls.divide(season_lbls.sum(axis=0), axis=1)    
    
    random_likelihood = 1/len(month_lbls)

    season_lbls2.loc['summer'] = season_lbls2.loc['summer'].apply(lambda x: x/(8*random_likelihood))
    season_lbls2.loc['winter'] = season_lbls2.loc['winter'].apply(lambda x: x/(4*random_likelihood))    
    
    return random_likelihood, season_lbls2