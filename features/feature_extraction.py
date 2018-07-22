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

from support import fdata_dir, writeLog, image_dir
from features.feature_socios import genS
from features.feature_ts import genX
from evaluation.evalClusters import bestLabels

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

def genF(experiment, year_start, year_end, drop_0, socios):

    X = genX([1994,2014], drop_0)
    Xdd = pd.DataFrame(X.sum(axis=1), columns=['DD']).reset_index()
    del X
    # Add cluster labels
    Xdd['k'] = bestLabels(experiment, n_best=1).values + 1
    # Add temporal features
    Xdd['year']=Xdd.date.dt.year
    Xdd['month']=Xdd.date.dt.month_name()
    Xdd['weekday']=Xdd.date.dt.weekday_name
    
    winter = ['May','June','July','August']
    work_week = ['Monday','Tuesday','Wednesday','Thursday']
    
    Xdd['season'] = Xdd.month.where(Xdd.month.isin(winter), 'summer')
    Xdd['season'] = Xdd.season.where(Xdd.season=='summer', 'winter')
    Xdd['daytype'] = Xdd.weekday.where(~Xdd.weekday.isin(work_week), 'weekday')
    
    S = genS(socios, year_start, year_end, 'csv').reset_index()
    Sdd = pd.concat([S, Xdd.groupby(['ProfileID']).DD.mean()], axis=1, join='inner').rename(columns={'DD':'ADD'})

    Xdd.drop(columns=['date','month','weekday','DD'], inplace=True)
        
    #merge socio-demographic, geographic and temporal features
    F = pd.merge(Xdd, Sdd, how='inner',on='ProfileID')
    del Xdd, S, Sdd
    
    monthly_consumption_bins = [5, 50, 150, 400, 600, 1200, 2500, 4000]
    daily_demand_bins = [x /30*1000/230 for x in monthly_consumption_bins]
    bin_labels = ['{0:.0f}-{1:.0f}'.format(x,y) for x, y in zip(daily_demand_bins[:-1], daily_demand_bins[1:])]
    F['ADD'] = pd.cut(F.ADD, daily_demand_bins, labels=bin_labels, right=False)
    F['ADD'] = F.ADD.where(~F.ADD.isna(), 0)
    F.iloc[:,1:] = F.iloc[:,1:].apply(lambda x: x.astype('category'))
    F.ADD.cat.reorder_categories([0]+bin_labels, ordered=True,inplace=True)
    
    for c in F.columns.drop(['ProfileID']):
        plotF(F, [c], socios)
    
    return F

def features2dict(data):      
    """            
    This function converts a dataframe into a dict formatted for use as evidence in libpgm BN inference.
    
    Unsure if this function has much use going forward ...19 July 2018
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