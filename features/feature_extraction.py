# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:59:37 2017

@author: Wiebke Toussaint
"""

import pandas as pd
import os
import numpy as np
import datetime as dt

from support import fdata_dir, writeLog
from features.feature_socios import genS
from features.feature_ts import genX
from evaluation.evalClusters import bestLabels

def genF(experiment, year_start, year_end, socios):
    
    X = genX([year_start,year_end])
    Xdd = pd.DataFrame(X.sum(axis=1), columns=['DD']).reset_index()
    del X
    # Add cluster labels
    Xdd['k'] = bestLabels(experiment, n_best=1).values + 1
    # Add temporal features
    Xdd['year']=Xdd.date.dt.year
    Xdd['month']=Xdd.date.dt.month_name()
    Xdd['weekday']=Xdd.date.dt.weekday_name
    Xdd.drop(columns='date', inplace=True)
    
    S = genS(socios, year_start, year_end, 'csv').reset_index()
    
    #merge socio-demographic, geographic and temporal features
    F = pd.merge(Xdd, S, how='inner',on='ProfileID')
    del Xdd, S
    
    monthly_consumption_bins = [5, 50, 100, 150, 200, 250, 300, 350, 400, 600, 900, 1200, 2500, 5000, 7000]
    daily_demand_bins = [x /30*1000/230 for x in monthly_consumption_bins]
    bin_labels = ['{0:.0f}-{1:.0f}'.format(x,y) for x, y in zip(daily_demand_bins[:-1], daily_demand_bins[1:])]
    F['DD'] = pd.cut(F.DD, daily_demand_bins, labels=bin_labels, right=False)
    F['DD'] = F.DD.where(~F.DD.isna(), 0)
    F.iloc[:,1:] = F.iloc[:,1:].apply(lambda x: x.astype('category'))
    
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