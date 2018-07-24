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
from glob import glob
import peakutils

import features.feature_ts as ts
from support import data_dir, results_dir

def getExperiments(exp_root):
    """
    Retrieve experiments with root name exp_root from the results directory. 
    Returns list of unique experiments with root exp_root.
    """
    
    exps = glob(os.path.join(data_dir,'cluster_results',exp_root + '*'))
    experiments = list(pd.Series([('_').join(x.split('/')[-1].split('_')[:-1]) for x in exps]).drop_duplicates())
    
    return experiments

def readResults():
    cluster_results = pd.read_csv('results/cluster_results.csv')
    cluster_results.drop_duplicates(subset=['dbi','mia','experiment_name','amd_bin'],keep='last',inplace=True)
    cluster_results = cluster_results[cluster_results.experiment_name != 'test']
    cluster_results['clusters'] = cluster_results.loc[:, 'n_clust'].where( 
            cluster_results['n_clust'] > 0,
            cluster_results['som_dim']**2)
    
    return cluster_results

def selectClusters(cluster_results, n_best, experiment='all' ):
    if experiment=='all':
        experiment_clusters = cluster_results
    else:
        experiment_clusters = cluster_results.loc[cluster_results.experiment_name.str.contains(experiment),:]
        
    experiment_clusters = experiment_clusters.groupby(['experiment_name','som_dim',
                                                       'n_clust']).mean().reset_index() 
    best_clusters = experiment_clusters.loc[experiment_clusters.all_scores>0,:].nsmallest(
            columns='all_scores',n=n_best).reset_index(drop=True).reindex(
                    ['experiment_name','som_dim','n_clust','dbi','mia','silhouette','all_scores'],axis=1)
        
    return best_clusters

def getLabels(experiment, drop_0=False, count_best=1):
    
    labels = feather.read_dataframe(os.path.join(data_dir, 'cluster_results', 
                                                 experiment+'_labels.feather')).iloc[:,:count_best]
    X = ts.genX([1994,2014], drop_0).reset_index()   
    exp =  experiment.split('_',1)[-1]
    for i in range(0, count_best):
        X[exp+str(i+1)] = labels.iloc[:,i]
    
    return X.set_index(['ProfileID','date'])

def bestLabels(experiment, n_best=1):
    X = getLabels(experiment, n_best)
    data = X.iloc[:,-n_best:]
    data.columns = ['0_'+str(l) for l in data.max()+1]
    del X #clear memory
    data.reset_index(inplace=True)
    data.date = data.date.astype(dt.date)#pd.to_datetime(data.date)
    data.ProfileID = data.ProfileID.astype('category')
    data.set_index(['ProfileID','date'], inplace=True)
    data.sort_index(level=['ProfileID','date'], inplace=True)
    
    return data

def getCentroids(best_clusters, n_best=1):

    best_experiments = list(best_clusters.experiment_name.unique())
    centroid_files = dict(zip(best_experiments,[e+'_centroids.csv' for e in best_experiments]))
    centroids = {}
    for k, v in centroid_files.items():
        centroids[k] = pd.read_csv(os.path.join(data_dir, 'cluster_results', v))
    
    best_centroids = pd.DataFrame()
    for row in best_clusters.itertuples():
        df = centroids[row.experiment_name]
        c = df.loc[(df.som_dim==row.som_dim)&(df.n_clust==row.n_clust),:]
        best_centroids = best_centroids.append(c)
    best_centroids.drop_duplicates(subset=['som_dim','n_clust','k','experiment_name'],keep='last',inplace=True)
    
    experiment_name, som_dim, n_clust = best_clusters.loc[n_best-1,['experiment_name','som_dim','n_clust']]    
    
    data = best_centroids.set_index(['experiment_name','som_dim','n_clust','k'])
    data.sort_index(level=['experiment_name','som_dim','n_clust'], inplace=True)    
    centroids = data.loc[(experiment_name, som_dim, n_clust), 
                         [str(i) for i in range(0,24)]].reset_index(drop=True)
    cluster_size = data.loc[(experiment_name, som_dim, n_clust), 'cluster_size'].reset_index(drop=True)
    meta = dict(experiment_name=experiment_name.split('_',1)[1], n_best=n_best)
    
    return centroids, cluster_size, meta

def realCentroids(experiment, n_best=1):
    X = getLabels(experiment, n_best)
    data = X.iloc[:,list(range(0,24))+[-1]]
    exp = data.columns[-1]
    centroids = data.groupby(exp).mean()
    cluster_size = data.groupby(exp)['0'].count()
    del X
    meta = dict(experiment_name=experiment.split('_',1)[1], n_best=n_best)
    
    return centroids, cluster_size, meta

def clusterColNames(data):    
    data.columns = ['Cluster '+str(x+1) for x in data.columns]
    return data

def consumptionError(experiment, compare='total', n_best=1):
    """
    Calculate error metrics for total daily consumption (compare=total) or peak daily consumption (compare=peak).
    Returns 
    mean absolute percentage error, 
    median absolute percentage error, 
    median log accuracy ratio (Q=predicted/actual)
    median symmetric accuracy
    """

    X = getLabels(experiment, n_best)
    centroids, cluster_size, meta = realCentroids(experiment, n_best)
    
    if compare == 'total':
        X_dd = pd.concat([X.iloc[:,list(range(0,24))].sum(axis=1), X.iloc[:,-1]], axis=1, keys=['DD','k'])
        cent_dd = centroids.sum(axis=1).rename_axis('k',0).reset_index(name='DD')
    elif compare == 'peak':
        X_dd = pd.concat([X.iloc[:,list(range(0,24))].max(axis=1), X.iloc[:,-1]], axis=1, keys=['DD','k'])
        cent_dd = centroids.max(axis=1).rename_axis('k',0).reset_index(name='DD')
    del X  

    X_dd['ae'] = 0
    X_dd['logq'] = 0
    for y in cent_dd.itertuples(): 
        X_dd.loc[X_dd.k==y[1],'ae'] = [abs(x-y[2]) for x in X_dd.loc[X_dd.k==y[1],'DD']]
        try:
            X_dd.loc[X_dd.k==y[1],'logq'] = [log(y[2]/x) for x in X_dd.loc[X_dd.k==y[1],'DD']]
        except:
            print('Zero values. Could not compute log(Q) for cluster', str(y[1]))
            X_dd.loc[X_dd.k==y[1],'logq'] = np.inf

    X_dd['ape'] = X_dd.ae/X_dd.DD
    X_dd['alogq'] = X_dd['logq'].map(lambda x: abs(x))
            
    mape = X_dd.groupby('k')['ape'].mean()*100
    mdape = X_dd.groupby('k')['ape'].agg(np.median)*100
    mdlq = X_dd.groupby('k')['logq'].agg(np.median)
    mdsyma = np.expm1(X_dd.groupby('k')['alogq'].agg(np.median))*100
    
    del X_dd
           
    return mape, mdape, mdlq, mdsyma

def centroidPeaks(experiment, n_best=1):
    centroids, cluster_size, meta = realCentroids(experiment, n_best)
    cent_peak = dict()
    for i in centroids.iterrows():
        h = peakutils.peak.indexes(i[1], thres=0.5, min_dist=1)
        val = centroids.iloc[i[0], h].values
        cent_peak[i[0]] = dict(zip(h,val))
        
    return cent_peak

def peakCoincidence(experiment, n_best=1):
    
    try:
        #get peakcoincidence from csv
        data=pd.read_csv(os.path.join(results_dir, 'peak_coincidence.csv'))
        peak_eval = data.loc[(data['experiment']==experiment)&(data['n_best']==n_best),:]
        peak_eval = peak_eval.drop_duplicates(subset=['k', 'experiment', 'n_best'], inplace=False, keep='last')
        if len(peak_eval) == 0:
            raise Exception
    except:
        X = getLabels(experiment, n_best)
        X2 = pd.concat([X.iloc[:,list(range(0,24))], X.iloc[:,-1]], axis=1)
        X2.columns = list(range(0,24))+['k']
        del X
        
        cent_peak = centroidPeaks(experiment, n_best)
    
        clusters = X2.iloc[:,-1].unique()
        clusters.sort()
        X_peak = dict()
        for c in clusters:
            X_k = X2.loc[X2.k == c]      
            X_k.drop(columns='k', inplace=True)
            peak_count = 0
            for i in X_k.iterrows():
                h = peakutils.peak.indexes(i[1], thres=0.5, min_dist=1)
                peak_count += len(set(cent_peak[c]).intersection(set(h)))
            X_peak[c] = peak_count / len(X_k)
            print('Mean peak coincidence computed for cluster',str(c))
    
        peak_eval = pd.DataFrame(list(X_peak.items()), columns=['k','mean_coincidence'])
        count_cent_peaks = [len(cent_peak[i].keys()) for i in cent_peak.keys()]
        peak_eval['coincidence_ratio'] = peak_eval.mean_coincidence/count_cent_peaks
        peak_eval['experiment'] = experiment
        peak_eval['n_best'] = n_best
        
        pcpath = os.path.join(results_dir, 'peak_coincidence.csv')
        if os.path.isfile(pcpath):
            peak_eval.to_csv(pcpath, mode='a', index=False, header=False)
        else:
            peak_eval.to_csv(pcpath, index=False)
        
        del X2    
    
    return peak_eval

def meanError(metric_vals):    
    err = metric_vals.where(~np.isinf(metric_vals)).mean()    
    return err

def demandCorr(experiment, compare='total', n_best=1):

    X = getLabels(experiment, n_best)
    if compare == 'total':
        data = pd.concat([X.iloc[:,list(range(0,24))].sum(axis=1), X.iloc[:,-1]], axis=1, keys=['DD','k'])
    elif compare == 'peak':
        data = pd.concat([X.iloc[:,list(range(0,24))].max(axis=1), X.iloc[:,-1]], axis=1, keys=['DD','k'])

    del X #clear memory
    
    data.reset_index(inplace=True)
    data.date = data.date.astype(dt.date)#pd.to_datetime(data.date)
    data.ProfileID = data.ProfileID.astype('category')
    
    #bin daily demand into 100 equally sized bins
    data['int100_bins']=pd.cut(data.loc[data.DD!=0,'DD'], bins = range(0,1000,10), 
        labels=np.arange(1, 100), include_lowest=False, right=True)
    data.int100_bins = data.int100_bins.cat.add_categories([0])
    data.int100_bins = data.int100_bins.cat.reorder_categories(range(0,100), ordered=True)
    data.loc[data.DD==0,'int100_bins'] = 0   
    
    #NB: use int100 for entropy calculation!
    int100_lbls = data.groupby(['k', data.int100_bins])['ProfileID'].count().unstack(level=0)
    int100_lbls = clusterColNames(int100_lbls)
    int100_likelihood = int100_lbls.divide(int100_lbls.sum(axis=0), axis=1)

    data['q100_bins'] = pd.qcut(data.loc[data.DD!=0,'DD'], q=99, labels=np.arange(1, 100))
    data.q100_bins = data.q100_bins.cat.add_categories([0])
    data.q100_bins = data.q100_bins.cat.reorder_categories(range(0,100), ordered=True)    
    data.loc[data.DD==0,'q100_bins'] = 0
    cats = data.groupby('q100_bins')['DD'].max().round(2)
    data.q100_bins.cat.categories = cats
    
    q100_lbls = data.groupby(['k', data.q100_bins])['ProfileID'].count().unstack(level=0)
    q100_lbls = clusterColNames(q100_lbls)
    q100_likelihood = q100_lbls.divide(q100_lbls.sum(axis=0), axis=1)
    
    return int100_likelihood, q100_likelihood

def weekdayCorr(label_data):

    if len(label_data.columns)>1:
        return('Too many columns to compute. Select 1 column only')
    else:
        label_data.columns = ['k']
    df = label_data.reset_index()
    weekday_lbls = df.groupby(['k',df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    weekday_lbls = clusterColNames(weekday_lbls)
    weekday_likelihood = weekday_lbls.divide(weekday_lbls.sum(axis=0), axis=1) # likelihood of assignment
    
    random_likelihood = 1/len(weekday_likelihood) # null hypothesis
    relative_likelihood = weekday_likelihood.divide(random_likelihood, axis=1)

    random_loglike = log(random_likelihood,2)#-random_likelihood*log(random_likelihood,2)    
    weekday_loglike = weekday_likelihood.applymap(lambda x : log(x,2))#-x*log(x,2)) 
    relative_loglike = weekday_loglike.divide(random_loglike, axis=1)
    
    return weekday_likelihood, relative_likelihood

def monthlyCorr(label_data):
    
    if len(label_data.columns)>1:
        return('Too many columns to compute. Select 1 column only')
    else:
        label_data.columns = ['k']
    df = label_data.reset_index()
    month_lbls = df.groupby(['k',df.date.dt.month])['ProfileID'].count().unstack(level=0)
    month_lbls = clusterColNames(month_lbls)
    month_likelihood = month_lbls.divide(month_lbls.sum(axis=0), axis=1)
    
    random_likelihood = 1/len(month_likelihood)    
    relative_likelihood = month_likelihood.divide(random_likelihood, axis=1)    
    
    return month_likelihood, relative_likelihood

def yearlyCorr(label_data):
    
    if len(label_data.columns)>1:
        return('Too many columns to compute. Select 1 column only')
    else:
        label_data.columns = ['k']
    df = label_data.reset_index()
    year_lbls = df.groupby(['k',df.date.dt.year])['ProfileID'].count().unstack(level=0)
    year_lbls = clusterColNames(year_lbls).T
    year_likelihood = year_lbls.divide(year_lbls.sum(axis=0), axis=1)    
    
    random_likelihood = 1/len(year_likelihood)    
    relative_likelihood = year_likelihood.divide(random_likelihood, axis=1)
    
    return year_likelihood, relative_likelihood, 

def daytypeCorr(label_data):
    
    if len(label_data.columns)>1:
        return('Too many columns to compute. Select 1 column only')
    else:
        label_data.columns = ['k']
    df = label_data.reset_index()
    weekday_lbls = df.groupby(['k',df.date.dt.weekday])['ProfileID'].count().unstack(level=0)
    weekday_lbls.set_axis(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], axis=0, inplace=True)
    daytype = weekday_lbls[weekday_lbls.index.isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])].sum(axis=0).to_frame('weekday').T
    daytype = daytype.append(weekday_lbls.loc[['Saturday','Sunday'], :])
    daytype_lbls = clusterColNames(daytype)
    daytype_likelihood = daytype_lbls.divide(daytype_lbls.sum(axis=0), axis=1)

    random_likelihood = [5/7, 1/7, 1/7]
    relative_likelihood = daytype_likelihood.divide(random_likelihood, axis=0)
   
    return daytype_likelihood, relative_likelihood

def seasonCorr(label_data):
    
    if len(label_data.columns)>1:
        return('Too many columns to compute. Select 1 column only')
    else:
        label_data.columns = ['k']
    df = label_data.reset_index()
    month_lbls = df.groupby(['k',df.date.dt.month])['ProfileID'].count().unstack(level=0)    
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

def householdEntropy(label_data):
    
    if len(label_data.columns)>1:
        return('Too many columns to compute. Select 1 column only')
    else:
        label_data.columns = ['k']
    df = label_data.reset_index()
    
    data = df.groupby(['ProfileID','k'])['date'].count().rename('day_count').reset_index()
    hh_lbls = data.pivot(index='ProfileID',columns='k',values='day_count')
    hh_likelihood = hh_lbls.divide(hh_lbls.sum(axis=1), axis=0)
    random_likelihood = 1/47
    
    cluster_entropy = hh_likelihood.applymap(lambda x : -x*log(x,2)).sum(axis=1)
    max_entropy = -random_likelihood*log(random_likelihood,2)*47
    
    return cluster_entropy, max_entropy

def monthlyHHE(lbls, S, month_ix):
    hhe, me = householdEntropy(lbls[lbls.date.dt.month==month_ix].set_index(['ProfileID','date']))
    Sent = pd.concat([S, (hhe/me)], axis=1, join='inner').rename(columns={0:'rele'})
    sg = Sent.groupby('monthly_income').aggregate({'rele':['mean','std']})
    return sg