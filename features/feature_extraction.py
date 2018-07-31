#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from support import writeLog, image_dir, data_dir, experiment_dir
from features.feature_socios import genS
from evaluation.eval_clusters import getLabels, getExpDetails

def plotF(F, columns, save_name=None):
    """
    Plots column category counts in F. Columns must be a list.
    """
    if save_name is None:
        dir_path = os.path.join(image_dir, 'experiment')
    else:
        dir_path = os.path.join(image_dir, 'experiment', save_name)
    os.makedirs(dir_path, exist_ok=True)
    
    data = F[['ProfileID']+columns]

    if len(columns) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        data.groupby(columns).ProfileID.count().plot('bar', 
                          title='Count of observations in ' + columns[0] + ' bins', figsize=(10, 6))
        plt.tight_layout()
        fig.savefig(os.path.join(dir_path, columns[0]+'.png'))
        plt.close()
    
    elif len(columns) > 1:
        fig, axes = plt.subplots(nrows=len(data.columns)-1, ncols=1)
        r = 0
        for c in data.columns.drop('ProfileID'):
            data.groupby(c).ProfileID.count().plot('bar', 
                        title='Count of observations in ' + c + ' bins', 
                        ax=axes[r], figsize=(10-len(data.columns)/4, len(data.columns)*4))
            r += 1
            plt.tight_layout()            

        fig.savefig(os.path.join(dir_path, save_name+'.png'))
        plt.close()
    
    return fig

def genF(experiment, socios, n_best=1, savefig=False):
    
    year_start, year_end, drop_0, prepro, exp_root = getExpDetails(experiment)
    
    kf_path = os.path.join(data_dir, 'cluster_evaluation','k_features', 
                           experiment+'_'+socios+'BEST'+str(n_best)+'.csv')
    
    if os.path.exists(kf_path) is True:
        F = pd.read_csv(kf_path)

    else:
        print('Extracting and creating feature data...')
        # Get cluster labels
        XL = getLabels(experiment, n_best)
        XL['DD'] = XL.iloc[:,list(range(0,24))].sum(axis=1)
        XL = XL.drop(columns=[str(i) for i in range(0,24)], axis=0).reset_index()
    
        # Add temporal features
        XL['year']=XL.date.dt.year
        XL['month']=XL.date.dt.month_name()
        XL['weekday']=XL.date.dt.weekday_name
        
        winter = ['May','June','July','August']
        work_week = ['Monday','Tuesday','Wednesday','Thursday']
        
        XL['season'] = XL.month.where(XL.month.isin(winter), 'summer')
        XL['season'] = XL.season.where(XL.season=='summer', 'winter')
        XL['daytype'] = XL.weekday.where(~XL.weekday.isin(work_week), 'weekday')
        
        S = genS(socios, year_start, year_end, 'feather').reset_index()
    
        XL.drop(columns=['date','month','weekday'], inplace=True)
            
        #merge socio-demographic, geographic and temporal features
        F = pd.merge(XL, S, how='inner',on='ProfileID')
        del XL, S
        
        monthly_consumption_bins = [5, 50, 150, 400, 600, 1200, 2500, 4000]
        daily_demand_bins = [x /30*1000/230 for x in monthly_consumption_bins]
        bin_labels = ['{0:.0f}-{1:.0f}'.format(x,y) for x, y in zip(daily_demand_bins[:-1], daily_demand_bins[1:])]
        F['DD'] = pd.cut(F.DD, daily_demand_bins, labels=bin_labels, right=False)
        F['DD'] = F.DD.where(~F.DD.isna(), 0)
        F['DD'] = F.DD.astype('category')
        F.DD.cat.reorder_categories([0]+bin_labels, ordered=True,inplace=True)
        
        F.drop('ProfileID', axis=1, inplace=True)
        F.to_csv(kf_path, index=False)
    
    if savefig is True:
        for c in F.columns.drop(['ProfileID']):
            plotF(F, [c], socios)
    
    return F

def features2dict(data):      
    """            
    This function converts a dataframe into a dict formatted for use as evidence in libpgm BN inference.
    
    Unsure if this function has much use going forward ...19 July 2018
    
    if drop_0 is False:
        f_path = os.path.join(data_dir, 'cluster_evaluation', 'k_features', 
                                  experiment+'_'+socios+str(year_start)+str(year_end)+'.feather')
    elif drop_0 is True:
        f_path = os.path.join(data_dir, 'cluster_evaluation', 'k_features', 
                                  experiment+'_'+socios+'drop0'+str(year_start)+str(year_end)+'.feather')  

    if os.path.exists(f_path) is True:
        F = feather.read_dataframe(f_path)
    """
    for c in data.columns:
        data[c].replace(np.nan, '', regex=True, inplace=True) #remove nan as BN inference cannot deal 
    featuredict = data.to_dict('index') 
    e = []
    for f in featuredict.values(): 
        d = dict()
        for k, v in f.items():
            if v is not str(''):
                d[k] = v
        e.append(d)  
    evidence = dict(zip(featuredict.keys(), e))
    
    return evidence

def checkFeatures(data, appliances):
    """
    This function error checks appliance features for records that indicate appliance usage but no ownership.
    """
    
    err = pd.DataFrame()
    for a in appliances:
        try:
            e = data.loc[(data[a]==0)&(data[a+'_use']>0), [a,a+'_use',a+'_broken']]
            print(e)
            err = err.append(e)
        except:
            pass
        
    return err